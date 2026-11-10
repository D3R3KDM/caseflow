"""Train the neural surrogate (engine/surrogate.py) on the simulated layouts of scripts/gen_surrogate_data.py.

Every cached layout gives one sample per case-fan duty (three, or one for the layout with no case fan). The test
layouts (every 7th random one) are never trained on; they measure how well the network does on layouts it has not
seen: field errors, the per-watt rises at the CPU and GPU intakes, and - what matters - the CPU and GPU temperatures
of the reference scenarios, solved once with the simulated bank and once with the network's.

Loss, on the air inside the case: squared error of the velocity (m/s / U0) and of log1p(rise / EPS_K), plus the
error of the four intake rises (CPU and GPU heat at the CPU and GPU intakes, in log1p(k / EPS_K)), which set the
part temperatures.

    python scripts/train_surrogate.py                      # all training layouts -> data/surrogate/caseflow_unet.pt
    python scripts/train_surrogate.py --frac 0.5 --tag half   # learning curve: half of them, same test layouts
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import bank as bk  # noqa: E402
from engine import scenario as sc  # noqa: E402
from engine.optimize import SCENARIOS  # noqa: E402
from engine.sim import build_model, load_library  # noqa: E402
from engine.surrogate import EPS_K, CaseUNet, build_from_code, count_params, decode, encode, target  # noqa: E402

OUT = ROOT / "data" / "surrogate"


def load_samples(rows: list[dict], lib: dict) -> dict:
    """Inputs, targets and intake rises of every level of every cached layout (float16 on the CPU)."""
    X, Y, P, meta = [], [], [], []
    for r in rows:
        b = bk.load(r["key"])
        if b is None:
            continue
        build = build_from_code(tuple(r["code"]))
        for i, (L, f) in enumerate(zip(b["summary"]["levels"], b["fields"])):
            m = build_model(build, lib, case_duty=L["duty"])
            X.append(torch.from_numpy(encode(m)).half())
            Y.append(torch.from_numpy(target(f["u"], f["kc"], f["kg"])).half())
            P.append(L["k_cpu_in"] + L["k_gpu_in"])            # [CPU, GPU heat at the CPU intake, same at the GPU intake]
            meta.append({"key": r["key"], "split": r["split"], "code": r["code"], "level": i, "duty": L["duty"]})
    m = build_model(build_from_code((1,) * 8), lib)
    probes = torch.from_numpy(np.stack([m["intake"]["cpu"], m["intake"]["gpu"]]))
    return {"X": torch.stack(X), "Y": torch.stack(Y), "P": torch.tensor(P, dtype=torch.float32), "meta": meta, "probes": probes}


def intake_rises(kc: torch.Tensor, kg: torch.Tensor, probes: torch.Tensor) -> torch.Tensor:
    """[B, 4]: CPU and GPU heat at the CPU intake, then at the GPU intake (the order of the bank's k_cpu_in + k_gpu_in)."""
    out = []
    for p in probes:                       # CPU intake, GPU intake
        w = p.float()
        n = w.sum()
        out += [(kc * w).sum((-3, -2, -1)) / n, (kg * w).sum((-3, -2, -1)) / n]
    return torch.stack(out, -1)


def losses(pred: torch.Tensor, y: torch.Tensor, x: torch.Tensor, p_true: torch.Tensor, probes: torch.Tensor) -> dict:
    pred = pred.float()
    inner = ((x[:, 0] < 0.5) & (x[:, 1] < 0.5)).unsqueeze(1).float()      # air inside the case
    n = inner.sum((2, 3, 4))
    se = (pred - y.float()) ** 2 * inner
    l_u = (se[:, :3].sum((2, 3, 4)) / n).sum(1).mean()
    l_k = (se[:, 3:].sum((2, 3, 4)) / n).sum(1).mean()
    _, kc, kg = decode(pred)
    p_pred = intake_rises(kc, kg, probes)
    l_p = ((torch.log1p(p_pred / EPS_K) - torch.log1p(p_true / EPS_K)) ** 2).mean()
    return {"u": l_u, "k": l_k, "probe": l_p}


@torch.no_grad()
def predict(net, X: torch.Tensor, idx: list[int], dev: str, batch: int = 8) -> torch.Tensor:
    outs = []
    for i in range(0, len(idx), batch):
        x = X[idx[i:i + batch]].to(dev).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            outs.append(net(x).float().cpu())
    return torch.cat(outs)


@torch.no_grad()
def field_metrics(net, data: dict, idx: list[int], dev: str) -> dict:
    """Mean over samples: velocity and rise RMSE in the case air, and the intake rises' absolute errors."""
    pred = predict(net, data["X"], idx, dev)
    probes = data["probes"]
    err_u, err_k, err_p, rel_p = [], [], [], []
    for j, i in enumerate(idx):
        x, y = data["X"][i].float(), data["Y"][i].float()
        inner = (x[0] < 0.5) & (x[1] < 0.5)
        u_p, kc_p, kg_p = decode(pred[j])
        u_t, kc_t, kg_t = decode(y)
        err_u.append(float(((u_p - u_t) ** 2).sum(0)[inner].mean().sqrt()))
        err_k.append(float(torch.cat([(kc_p - kc_t)[inner], (kg_p - kg_t)[inner]]).pow(2).mean().sqrt()))
        p = intake_rises(kc_p * inner, kg_p * inner, probes)
        err_p.append((p - data["P"][i]).abs().tolist())
        rel_p.append(float((p - data["P"][i]).abs().sum() / data["P"][i].abs().sum()))
    e = np.array(err_p)
    return {"u_rmse_ms": float(np.mean(err_u)), "k_rmse_kw": float(np.mean(err_k)), "intake_mae_kw": float(e.mean()),
            "intake_max_kw": float(e.max()), "intake_rel": float(np.mean(rel_p))}


def mean_rises(rows: list[dict]) -> list[dict]:
    """Baseline: the training layouts' average per-watt rises at each of the three duties."""
    acc = [{k: np.zeros(2) for k in sc.RISE_KEYS} for _ in bk.LEVELS]
    n = 0
    for r in rows:
        js = bk.CACHE / f"{r['key']}.json"
        lv = json.loads(js.read_text())["levels"] if js.exists() else []
        if len(lv) != len(bk.LEVELS):
            continue
        n += 1
        for a, L in zip(acc, lv):
            for k in sc.RISE_KEYS:
                a[k] += np.array(L[k])
    return [{k: (v / max(n, 1)).tolist() for k, v in a.items()} for a in acc]


@torch.no_grad()
def temperature_errors(net, data: dict, rows: list[dict], lib: dict, dev: str, baseline: list[dict] | None = None) -> dict:
    """CPU / GPU temperature of each test layout in each reference scenario: simulated bank vs network bank (and vs
    the baseline that predicts the training layouts' average rises for every layout)."""
    presets = {p["id"]: p for p in lib["defaults"]["scenarios"]}
    by_key = {}
    for i, mt in enumerate(data["meta"]):
        by_key.setdefault(mt["key"], []).append(i)
    per_layout, errs, errs_base = [], [], []
    for r in rows:
        if r["key"] not in by_key:
            continue
        b = bk.load(r["key"])
        build = build_from_code(tuple(r["code"]))
        idx = by_key[r["key"]]
        pred = predict(net, data["X"], idx, dev)
        ai = copy.deepcopy(b["summary"])
        for j, L in enumerate(ai["levels"]):
            m = build_model(build, lib, case_duty=L["duty"])
            u, kc, kg = decode(pred[j])
            inner = torch.from_numpy(~(m["solid"] | m["reservoir"]))
            s = bk.level_summary(L["duty"], m, u, kc * inner, kg * inner, None)
            for k in sc.RISE_KEYS:
                L[k] = s[k]
        base = None
        if baseline and len(ai["levels"]) == len(baseline):
            base = copy.deepcopy(b["summary"])
            for L, mr in zip(base["levels"], baseline):
                L.update(mr)
        su = sc.setup_for(build, lib)
        rows_s = []
        for pid, room in SCENARIOS:
            pc, pg = sc.load_watts(su, presets[pid]["cpu"], presets[pid]["gpu"])
            t = sc.solve(b["summary"], su, room, pc, pg)
            a = sc.solve(ai, su, room, pc, pg)
            d = (a["cpu"]["temp_c"] - t["cpu"]["temp_c"], a["gpu"]["temp_c"] - t["gpu"]["temp_c"])
            errs.append(d)
            if base:
                z = sc.solve(base, su, room, pc, pg)
                errs_base.append((z["cpu"]["temp_c"] - t["cpu"]["temp_c"], z["gpu"]["temp_c"] - t["gpu"]["temp_c"]))
            rows_s.append({"scenario": pid, "room_c": room, "cpu_sim": t["cpu"]["temp_c"], "cpu_ai": a["cpu"]["temp_c"],
                           "gpu_sim": t["gpu"]["temp_c"], "gpu_ai": a["gpu"]["temp_c"]})
        per_layout.append({"code": "".join("-io"[c] for c in r["code"]), "key": r["key"], "scenarios": rows_s})
    e = np.abs(np.array(errs))
    out = {"cpu_mae_c": float(e[:, 0].mean()), "gpu_mae_c": float(e[:, 1].mean()), "cpu_max_c": float(e[:, 0].max()),
           "gpu_max_c": float(e[:, 1].max()), "n_layouts": len(per_layout), "n_cases": len(errs), "layouts": per_layout}
    if errs_base:
        eb = np.abs(np.array(errs_base))
        out["baseline"] = {"what": "every layout gets the training layouts' average rises", "cpu_mae_c": float(eb[:, 0].mean()),
                           "gpu_mae_c": float(eb[:, 1].mean()), "cpu_max_c": float(eb[:, 0].max()), "gpu_max_c": float(eb[:, 1].max())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--w-probe", type=float, default=1.0)
    ap.add_argument("--frac", type=float, default=1.0, help="share of the training layouts to use (learning curve)")
    ap.add_argument("--max-layouts", type=int, default=0, help="smoke test: use only this many cached layouts")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--tag", default="", help="suffix of the output files")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lib = load_library()
    ds = json.loads((OUT / "dataset.json").read_text())
    rows = [r for r in ds["rows"] if (bk.CACHE / f"{r['key']}.json").exists()]     # the .json is written last
    if args.max_layouts:
        rows = rows[:args.max_layouts]
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    if args.max_layouts and not test_rows:            # smoke test before any test layout is simulated
        train_rows, test_rows = train_rows[:-2], [dict(r, split="test") for r in train_rows[-2:]]
    if args.frac < 1.0:
        train_rows = random.Random(1).sample(train_rows, max(1, round(args.frac * len(train_rows))))
    t0 = time.time()
    data = load_samples(train_rows + test_rows, lib)
    tr = [i for i, mt in enumerate(data["meta"]) if mt["split"] == "train"]
    te = [i for i, mt in enumerate(data["meta"]) if mt["split"] == "test"]
    print(f"data: {len(train_rows)} train layouts ({len(tr)} samples), {len(test_rows)} test layouts ({len(te)} samples), "
          f"grid {tuple(data['X'].shape[2:])}, loaded in {time.time() - t0:.0f} s", flush=True)

    net = CaseUNet(base=args.base).to(dev)
    print(f"CaseUNet base {args.base}: {count_params(net) / 1e6:.2f} M parameters", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(tr) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    probes = data["probes"].to(dev)
    hist = []
    rng = np.random.default_rng(0)
    t_train = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = rng.permutation(tr)
        acc = {"u": 0.0, "k": 0.0, "probe": 0.0}
        nb = 0
        te0 = time.time()
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch].tolist()
            x = data["X"][idx].to(dev, non_blocking=True).float()
            y = data["Y"][idx].to(dev, non_blocking=True)
            p = data["P"][idx].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                pred = net(x)
            ls = losses(pred, y, x, p, probes)
            loss = ls["u"] + ls["k"] + args.w_probe * ls["probe"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            for k in acc:
                acc[k] += float(ls[k].detach())
            nb += 1
        h = {"epoch": ep, "lr": sched.get_last_lr()[0], "seconds": time.time() - te0, **{f"train_{k}": v / nb for k, v in acc.items()}}
        if te and (ep % args.eval_every == 0 or ep == args.epochs):
            net.eval()
            h["test"] = field_metrics(net, data, te, dev)
            h["train_fields"] = field_metrics(net, data, tr[::max(1, len(tr) // 48)], dev)
        hist.append(h)
        msg = f"[ep {ep:3d}] loss u {h['train_u']:.4f} k {h['train_k']:.4f} probe {h['train_probe']:.4f}  {h['seconds']:.1f} s"
        if "test" in h:
            t = h["test"]
            msg += (f" | test: u {t['u_rmse_ms']:.3f} m/s, rise {t['k_rmse_kw'] * 1000:.2f} mK/W, intake {t['intake_mae_kw'] * 1000:.2f}"
                    f" mK/W (max {t['intake_max_kw'] * 1000:.1f}, {t['intake_rel'] * 100:.1f} %)")
        print(msg, flush=True)

    net.eval()
    temps = temperature_errors(net, data, test_rows, lib, dev, mean_rises(train_rows)) if te else None
    final = {"test": hist[-1].get("test"), "train": hist[-1].get("train_fields"), "temps": temps}
    if temps:
        print(f"test temperatures ({temps['n_layouts']} layouts x {len(SCENARIOS)} scenarios): CPU MAE {temps['cpu_mae_c']:.2f} "
              f"(max {temps['cpu_max_c']:.2f}) degC, GPU MAE {temps['gpu_mae_c']:.2f} (max {temps['gpu_max_c']:.2f}) degC", flush=True)
        if "baseline" in temps:
            z = temps["baseline"]
            print(f"  baseline (average rises of the training layouts): CPU MAE {z['cpu_mae_c']:.2f} (max {z['cpu_max_c']:.2f}), "
                  f"GPU MAE {z['gpu_mae_c']:.2f} (max {z['gpu_max_c']:.2f}) degC", flush=True)
    name = f"caseflow_unet{('_' + args.tag) if args.tag else ''}"
    info = {"name": name, "config": net.config, "params": count_params(net), "eps_k": EPS_K,
            "trained": {"layouts": len(train_rows), "samples": len(tr), "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                        "w_probe": args.w_probe, "minutes": (time.time() - t_train) / 60, "date": time.strftime("%Y-%m-%d")},
            "test": None if not temps else {"layouts": temps["n_layouts"], "cpu_mae_c": temps["cpu_mae_c"], "gpu_mae_c": temps["gpu_mae_c"],
                                             "cpu_max_c": temps["cpu_max_c"], "gpu_max_c": temps["gpu_max_c"], **(final["test"] or {})},
            "train_keys": sorted({mt["key"] for mt in data["meta"] if mt["split"] == "train"}),
            "test_keys": sorted({mt["key"] for mt in data["meta"] if mt["split"] == "test"})}
    if not args.max_layouts:
        torch.save({**info, "state": {k: v.half() for k, v in net.state_dict().items()}}, OUT / f"{name}.pt")
        (OUT / f"{name}_history.json").write_text(json.dumps(hist, indent=1))
        (OUT / f"{name}_eval.json").write_text(json.dumps({**{k: v for k, v in info.items() if k not in ("train_keys", "test_keys")},
                                                           "final": final}, indent=1))
    print(f"TRAIN DONE: {name}, {(time.time() - t_train) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
