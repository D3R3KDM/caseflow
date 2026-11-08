"""Detect what this PC can tell us without installing drivers: CPU / board / RAM (CIM), GPU sensors (nvidia-smi),
CPU package power (Windows' Energy Meter counters, i.e. RAPL), and CPU temperature only if LibreHardwareMonitor's
web server is running. Case-fan rpm and CPU temperature live in the board's Super I/O chip, which Windows does not
expose without a kernel driver."""
from __future__ import annotations

import json
import shutil
import subprocess
import time


def _ps(cmd: str):
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=20)
        return json.loads(out.stdout) if out.stdout.strip() else None
    except Exception:
        return None


def gpu_sensors() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    q = "name,temperature.gpu,fan.speed,power.draw,power.limit,utilization.gpu,clocks.gr"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]

        def num(x):
            try:
                return float(x)
            except ValueError:
                return None

        gpus.append({"name": f[0], "temp_c": num(f[1]), "fan_pct": num(f[2]), "power_w": num(f[3]),
                     "power_limit_w": num(f[4]), "util_pct": num(f[5]), "clock_mhz": num(f[6])})
    return gpus


class CpuPower:
    """CPU package power in W from Windows' built-in Energy Meter performance counters (AMD / Intel RAPL), read
    through pdh.dll. No driver or package needed. The counter is a rate: the first read after creation is None,
    later reads give the average power since the previous read. None when the counter does not exist."""

    PATH = r"\Energy Meter(RAPL_Package0_PKG)\Power"        # milliwatts

    def __init__(self):
        self._pdh = None
        try:
            import ctypes
            from ctypes import wintypes

            class Fmt(ctypes.Structure):
                _fields_ = [("CStatus", wintypes.DWORD), ("value", ctypes.c_double)]

            pdh = ctypes.WinDLL("pdh")
            pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
            pdh.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t,
                                                  ctypes.POINTER(wintypes.HANDLE)]
            pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
            pdh.PdhGetFormattedCounterValue.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                                        ctypes.POINTER(Fmt)]
            for f in (pdh.PdhOpenQueryW, pdh.PdhAddEnglishCounterW, pdh.PdhCollectQueryData,
                      pdh.PdhGetFormattedCounterValue):
                f.restype = ctypes.c_uint32
            q, c = wintypes.HANDLE(), wintypes.HANDLE()
            if pdh.PdhOpenQueryW(None, 0, ctypes.byref(q)) or pdh.PdhAddEnglishCounterW(q, self.PATH, 0, ctypes.byref(c)):
                return
            pdh.PdhCollectQueryData(q)
            self._pdh, self._q, self._c, self._fmt, self._ct = pdh, q, c, Fmt, ctypes
            self._t = time.monotonic()
        except Exception:                                         # not Windows, or no RAPL counters
            self._pdh = None

    def read(self) -> float | None:
        if self._pdh is None or time.monotonic() - self._t < 0.25:   # too short an interval to average over
            return None
        self._t = time.monotonic()
        if self._pdh.PdhCollectQueryData(self._q):
            return None
        v = self._fmt()
        if self._pdh.PdhGetFormattedCounterValue(self._c, 0x200 | 0x8000, None, self._ct.byref(v)) or v.CStatus > 1:
            return None                                           # 0x200 double, 0x8000 no cap; status 0/1 = valid
        return v.value / 1000.0


def cpu_temp_lhm() -> float | None:
    """LibreHardwareMonitor web server (Options -> Remote Web Server, port 8085), if the user runs it."""
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:8085/data.json", timeout=1.5) as r:
            data = json.loads(r.read())
    except Exception:
        return None
    best = None

    def walk(node, path):
        nonlocal best
        txt = node.get("Text", "")
        val = node.get("Value", "")
        if "°C" in str(val) and ("Package" in txt or "Tctl" in txt or "Tdie" in txt) and "CPU" in " ".join(path + [txt]).upper() + "CPU":
            try:
                best = float(str(val).replace("°C", "").replace(",", ".").strip())
            except ValueError:
                pass
        for ch in node.get("Children", []):
            walk(ch, path + [txt])

    walk(data, [])
    return best


def detect() -> dict:
    cpu = _ps("Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress") or {}
    board = _ps("Get-CimInstance Win32_BaseBoard | Select-Object Manufacturer,Product | ConvertTo-Json -Compress") or {}
    cs = _ps("Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | ConvertTo-Json -Compress") or {}
    fans = _ps("(Get-CimInstance Win32_Fan | Measure-Object).Count")
    gpus = gpu_sensors()
    cpu_t = cpu_temp_lhm()
    return {
        "cpu": {"name": (cpu.get("Name") or "").strip(), "cores": cpu.get("NumberOfCores"), "threads": cpu.get("NumberOfLogicalProcessors"),
                "max_mhz": cpu.get("MaxClockSpeed")},
        "board": f"{board.get('Manufacturer', '')} {board.get('Product', '')}".strip(),
        "ram_gb": round((cs.get("TotalPhysicalMemory") or 0) / 2 ** 30),
        "gpus": gpus,
        "cpu_temp_c": cpu_t,
        "cpu_temp_source": "LibreHardwareMonitor" if cpu_t is not None else None,
        "case_fans_detected": int(fans or 0),
        "notes": [
            "Case-fan rpm is in the board's Super I/O chip, which Windows does not expose without a sensor driver: "
            "set the fan speeds in Configure (Armoury Crate or the BIOS shows the rpm).",
            "CPU temperature needs a sensor driver; run LibreHardwareMonitor with its web server (port 8085) or type it in."
            if cpu_t is None else "CPU temperature read from LibreHardwareMonitor.",
        ],
    }


class SensorLog:
    """Samples the GPU (nvidia-smi) and CPU utilisation every few seconds in a background thread and keeps the
    last half hour, so a gaming session can be summarised afterwards: real GPU power, temperature, fan speed and
    utilisation under load, instead of one instant reading.

    CPU package power comes from Windows' Energy Meter counters (see CpuPower).

    busy: optional callable, True while CaseFlow itself runs a simulation on the GPU. Those samples are kept but
    left out of the summary, so the simulation's own GPU load is not mistaken for a game."""

    def __init__(self, period_s: float = 2.0, keep_min: float = 30.0, busy=None):
        import collections
        import threading

        self.period = period_s
        self.busy = busy
        self.buf: collections.deque = collections.deque(maxlen=int(keep_min * 60 / period_s))
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "SensorLog":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        import time

        try:
            import psutil
            psutil.cpu_percent(interval=None)                     # prime the counter
        except Exception:                                         # pragma: no cover - psutil missing
            psutil = None
        cpu_w = CpuPower()                                        # created in this thread, used only here
        while not self._stop.wait(self.period):
            g = gpu_sensors()
            row = {"t": time.time(), "cpu_util": psutil.cpu_percent(interval=None) if psutil else None,
                   "cpu_power_w": cpu_w.read(), "sim": bool(self.busy and self.busy())}
            if g:
                row.update({f"gpu_{k}": v for k, v in g[0].items() if k != "name"})
            with self.lock:
                self.buf.append(row)

    def latest(self) -> dict:
        with self.lock:
            return dict(self.buf[-1]) if self.buf else {}

    def summary(self, minutes: float = 10.0, min_gpu_w: float = 150.0) -> dict:
        """Averages over the last `minutes`; 'load' = only the samples where the GPU drew at least min_gpu_w.
        Samples taken while CaseFlow was simulating are left out (counted in sim_seconds)."""
        import time

        now = time.time()
        with self.lock:
            recent = [r for r in self.buf if now - r["t"] <= minutes * 60]
        rows = [r for r in recent if not r.get("sim")]

        def avg(rs, k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return sum(v) / len(v) if v else None

        loaded = [r for r in rows if (r.get("gpu_power_w") or 0) >= min_gpu_w]
        keys = ("gpu_temp_c", "gpu_power_w", "gpu_fan_pct", "gpu_util_pct", "gpu_clock_mhz", "cpu_util", "cpu_power_w")
        return {
            "minutes": minutes, "samples": len(rows), "period_s": self.period, "min_gpu_w": min_gpu_w,
            "load_seconds": len(loaded) * self.period, "sim_seconds": (len(recent) - len(rows)) * self.period,
            "all": {k: avg(rows, k) for k in keys},
            "load": {k: avg(loaded, k) for k in keys} if loaded else None,
            "gpu_temp_max": max((r.get("gpu_temp_c") or 0) for r in rows) if rows else None,
        }
