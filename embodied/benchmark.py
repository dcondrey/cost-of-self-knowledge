"""Portable cross-substrate benchmark: the real cost of self-introspection.

Runs on any machine (macOS / Linux / Windows; CI, Modal, Colab, laptop) and measures, in the
machine's own CPU-seconds, the cost of WORK vs the cost of self-observation at three modalities:
  - shallow_inproc : a cheap in-process self-read (load average)        -> expected ~free
  - proc_inproc    : richer in-process self-read (/proc, thermal files)  -> Linux; middle
  - deep_subprocess: rich self-telemetry via a spawned tool (the body)   -> expected expensive
Reports look_frac = cost_per_look / (cost_per_look + cost_per_work_unit) per modality, with repeats
for error bars, tagged with the machine identity. Emits JSON to stdout and benchmark_result.json.
The cross-machine claim: on EVERY real substrate, shallow introspection is cheap and deep bodily
self-telemetry is dear - and the cheap/dear split straddles the break-even price of self-knowledge.
"""
import glob
import json
import os
import platform
import statistics
import subprocess
import sys
import time

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

REPS = 9


def cpu_seconds():
    """Process CPU seconds incl. subprocess children (so subprocess looks are charged honestly)."""
    return sum(os.times()[:4])


def machine_id():
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "system": platform.system(), "cores": os.cpu_count() or 0,
            "python": platform.python_version(), "have_numpy": HAVE_NUMPY}
    try:
        if platform.system() == "Linux":
            for ln in open("/proc/cpuinfo"):
                if "model name" in ln:
                    info["cpu"] = ln.split(":", 1)[1].strip()
                    break
        elif platform.system() == "Darwin":
            info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                         capture_output=True, text=True, timeout=5).stdout.strip()
        else:
            info["cpu"] = platform.processor()
    except (OSError, subprocess.SubprocessError):
        info["cpu"] = platform.processor()
    info.setdefault("cpu", "")
    info["env"] = ("colab" if "COLAB_GPU" in os.environ else
                   "github_ci" if "GITHUB_ACTIONS" in os.environ else
                   "modal" if "MODAL_TASK_ID" in os.environ else "local")
    return info


def work_unit():
    if HAVE_NUMPY:
        a = np.random.rand(256, 256)
        float((a @ a).sum())
    else:
        x = 1.0
        for _ in range(200_000):
            x = (x * 1.0000001 + 1.0) % 1e6


def shallow_look():
    os.getloadavg()                          # cheap in-process self-read (raises on Windows)


def proc_look():
    for p in ("/proc/stat", "/proc/meminfo"):
        with open(p) as f:
            f.read()
    for f in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(f) as fh:
                fh.read()
        except OSError:
            pass


def _deep_cmd():
    s = platform.system()
    if s == "Linux":
        return ["sh", "-c", "cat /proc/stat /proc/meminfo /sys/class/thermal/thermal_zone*/temp 2>/dev/null"]
    if s == "Darwin":
        return ["sh", "-c", "sysctl -n vm.loadavg >/dev/null; vm_stat >/dev/null; ps -A -o %cpu= >/dev/null"]
    if s == "Windows":
        return ["cmd", "/c", "wmic cpu get loadpercentage"]
    return ["sh", "-c", "uptime"]


def deep_look():
    subprocess.run(_deep_cmd(), capture_output=True, timeout=15)


def measure(fn, n, warm=3):
    for _ in range(warm):
        fn()
    c0 = cpu_seconds()
    for _ in range(n):
        fn()
    return (cpu_seconds() - c0) / n


def main():
    work = [measure(work_unit, 300) for _ in range(REPS)]
    wmean = statistics.mean(work)
    result = {"machine": machine_id(), "reps": REPS,
              "work_unit_s_mean": wmean, "modalities": {}}
    for name, fn, n in (("shallow_inproc", shallow_look, 8000),
                        ("proc_inproc", proc_look, 2000),
                        ("deep_subprocess", deep_look, 60)):
        try:
            fn()                              # availability probe
        except (OSError, ValueError, subprocess.SubprocessError, AttributeError) as e:
            result["modalities"][name] = {"available": False, "reason": type(e).__name__}
            continue
        costs = [measure(fn, n) for _ in range(REPS)]
        fracs = [c / (c + wmean) for c in costs]
        result["modalities"][name] = {
            "available": True,
            "cost_s_mean": statistics.mean(costs),
            "look_frac_mean": statistics.mean(fracs),
            "look_frac_std": statistics.pstdev(fracs) if len(fracs) > 1 else 0.0,
        }
    out = json.dumps(result, indent=2)
    sys.stdout.write(out + "\n")
    with open("benchmark_result.json", "w") as f:
        f.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
