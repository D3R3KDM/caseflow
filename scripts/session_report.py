"""What did the PC actually do during a gaming session?

Joins two logs by time and averages them over the stretch where the GPU was under load:
- CaseFlow's own sensor log (logs/sensors_YYYYMMDD.csv, every 2 s): CPU and GPU power, GPU temperature and fan %;
- a Corsair iCUE sensor log (Settings -> Sensor Logging, saved to logs/icue/): motherboard fan headers (rpm),
  CPU package temperature, GPU fan rpm, board temperatures.

    python scripts/session_report.py                 # newest iCUE log in logs/icue, GPU >= 150 W counts as load
    python scripts/session_report.py --min-gpu-w 200 --icue logs/icue/corsair_cue_20261108_19_38_29.csv
    python scripts/session_report.py --json out.json  # also write the averages as JSON
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = re.compile(r"^\s*(-?[\d.]+)\s*(RPM|°C|%|W|V|MHz)?\s*$", re.I)
STAMP = re.compile(r"(\d+)/(\d+)/(\d{4})\s+(\d+):(\d+):(\d+)\s*([AP]M)?", re.I)


def icue_time(text: str, file_date: tuple[int, int, int] | None) -> float | None:
    """iCUE writes e.g. '8/11/2026 19:38:37 PM' (day/month here, 24 h with a stray AM/PM). The file name carries the
    start date (corsair_cue_YYYYMMDD_...), which settles day/month order."""
    m = STAMP.search(text)
    if not m:
        return None
    a, b, y, hh, mm, ss, ap = int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6]), (m[7] or "").upper()
    if hh < 12 and ap == "PM":
        hh += 12
    elif hh == 12 and ap == "AM":
        hh = 0
    day, mon = a, b
    if file_date and (file_date[1], file_date[2]) == (a, b):      # month/day order
        mon, day = a, b
    return time.mktime((y, mon, day, hh, mm, ss, 0, 0, -1))


def read_icue(path: Path) -> tuple[list[str], list[dict]]:
    fd = re.search(r"(\d{4})(\d{2})(\d{2})", path.name)
    file_date = tuple(int(x) for x in fd.groups()) if fd else None
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        r = csv.reader(fh)
        head = next(r)
        cols = [h.strip() for h in head[1:]]
        for rec in r:
            if not rec:
                continue
            t = icue_time(rec[0], file_date)
            if t is None:
                continue
            row = {"t": t}
            for name, v in zip(cols, rec[1:]):
                m = UNIT.match(v)
                if m:
                    row[name] = float(m[1])
            rows.append(row)
    return cols, rows


def read_sensors(t0: float, t1: float) -> list[dict]:
    """CaseFlow's sensor log for the days the session touches."""
    days = {time.strftime("%Y%m%d", time.localtime(t)) for t in (t0, t1)}
    rows = []
    for d in sorted(days):
        f = ROOT / "logs" / f"sensors_{d}.csv"
        if not f.exists():
            continue
        with f.open(encoding="utf-8", newline="") as fh:
            for rec in csv.DictReader(fh):
                t = float(rec["t"])
                if t0 - 5 <= t <= t1 + 5:
                    rows.append({k: (float(v) if v not in ("", None) and k != "time" else v) for k, v in rec.items()})
    return rows


def nearest(sens: list[dict], t: float, tol: float = 2.5) -> dict | None:
    lo, hi = 0, len(sens)
    while lo < hi:
        mid = (lo + hi) // 2
        if sens[mid]["t"] < t:
            lo = mid + 1
        else:
            hi = mid
    best = min((sens[i] for i in (lo - 1, lo) if 0 <= i < len(sens)), key=lambda r: abs(r["t"] - t), default=None)
    return best if best is not None and abs(best["t"] - t) <= tol else None


def stats(vals: list[float]) -> dict | None:
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return {"mean": statistics.fmean(v), "min": min(v), "max": max(v), "n": len(v)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--icue", type=Path, default=None)
    ap.add_argument("--min-gpu-w", type=float, default=150.0)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    icue_path = a.icue or max((ROOT / "logs" / "icue").glob("corsair_cue_*.csv"), key=lambda p: p.stat().st_mtime)
    cols, irows = read_icue(icue_path)
    if not irows:
        print("no rows in", icue_path)
        return 1
    t0, t1 = irows[0]["t"], irows[-1]["t"]
    sens = sorted(read_sensors(t0, t1), key=lambda r: r["t"])
    joined = []
    for r in irows:
        s = nearest(sens, r["t"])
        joined.append({**r, **({f"cf_{k}": v for k, v in s.items() if k not in ("t", "time")} if s else {})})
    load = [r for r in joined if (r.get("cf_gpu_power_w") or 0) >= a.min_gpu_w and not r.get("cf_sim")]
    idle = [r for r in joined if r.get("cf_gpu_power_w") is not None and r["cf_gpu_power_w"] < 60 and not r.get("cf_sim")]
    span = lambda rs: (len(rs) * (t1 - t0) / max(len(joined) - 1, 1)) / 60 if rs else 0.0   # minutes
    out = {"icue_file": icue_path.name, "start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
           "end": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)), "rows": len(joined),
           "matched_with_caseflow_log": sum(1 for r in joined if "cf_gpu_power_w" in r),
           "min_gpu_w": a.min_gpu_w, "minutes": (t1 - t0) / 60, "load_minutes": span(load), "idle_minutes": span(idle),
           "all": {}, "load": {}, "idle": {}}
    keys = cols + ["cf_cpu_power_w", "cf_cpu_util", "cf_gpu_power_w", "cf_gpu_temp_c", "cf_gpu_fan_pct", "cf_gpu_util_pct",
                   "cf_gpu_clock_mhz"]
    for part, rs in (("all", joined), ("load", load), ("idle", idle)):
        for k in keys:
            st = stats([r.get(k) for r in rs])
            if st:
                out[part][k] = st
    print(f"iCUE log {icue_path.name}: {out['start']} -> {out['end']}, {len(joined)} rows, "
          f"{out['matched_with_caseflow_log']} matched to CaseFlow's log")
    print(f"GPU >= {a.min_gpu_w:.0f} W for {out['load_minutes']:.1f} min; idle (< 60 W) for {out['idle_minutes']:.1f} min\n")
    print(f"{'sensor':28s} {'all mean':>9s} {'min':>8s} {'max':>8s}   {'load mean':>9s} {'idle mean':>9s}")
    for k in keys:
        A, L, I = out["all"].get(k), out["load"].get(k), out["idle"].get(k)
        if not A:
            continue
        name = k.replace("cf_", "CaseFlow ")
        f = lambda x, key: f"{x[key]:8.1f}" if x else f"{'-':>8s}"
        print(f"{name:28s} {f(A, 'mean'):>9s} {f(A, 'min')} {f(A, 'max')}   {f(L, 'mean'):>9s} {f(I, 'mean'):>9s}")
    if a.json:
        a.json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\nwrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
