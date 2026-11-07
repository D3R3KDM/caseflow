"""Detect what this PC can tell us without drivers: CPU / board / RAM (CIM), GPU sensors (nvidia-smi),
and CPU temperature only if LibreHardwareMonitor's web server is running (Windows does not expose it)."""
from __future__ import annotations

import json
import shutil
import subprocess


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
            "Case fans are not visible to Windows on most boards: set them in the fan editor.",
            "CPU temperature needs a sensor driver; run LibreHardwareMonitor with its web server (port 8085) or type it in."
            if cpu_t is None else "CPU temperature read from LibreHardwareMonitor.",
        ],
    }
