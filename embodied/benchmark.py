"""Portable cross-substrate benchmark: the real cost of self-introspection.

Runs on any machine (macOS / Linux / Windows / iOS a-shell; CI, Modal, Colab, Kaggle, laptop, phone)
and measures, in the machine's own CPU-seconds, two SUBSTRATE-INVARIANT raw facts:

  1. cost_per_look (seconds) for self-observation at three modalities:
       shallow_inproc  : a cheap in-process self-read (load average)        -> expected ~free
       proc_inproc     : richer in-process self-read (/proc, thermal files)  -> Linux; middle
       deep_subprocess : rich self-telemetry via a spawned tool (the body)   -> expected expensive
  2. cost_per_work (seconds) across a SWEEP of work-unit sizes.

look_frac = cost_per_look / (cost_per_look + cost_per_work) is then a DERIVED quantity that depends
on the work scale; reporting raw costs + the sweep makes that dependence explicit instead of baking
in one arbitrary work unit. The cross-substrate claim: on every real substrate shallow introspection
is ~free while deep bodily self-telemetry costs a large fraction of useful work, and there is a
critical work-unit scale below which knowing your own body is not worth it.

Emits JSON to stdout and benchmark_result.json. Only dependency is numpy (falls back to pure Python).
"""
import glob
import json
import os
import platform
import statistics
import subprocess
import sys

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

REPS = 9
WORK_LEVELS = [64, 128, 256, 512] if HAVE_NUMPY else [50_000, 100_000, 200_000, 400_000]


def cpu_seconds():
    """Process CPU seconds incl. subprocess children (so subprocess looks are charged honestly)."""
    return sum(os.times()[:4])


def machine_id():
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "system": platform.system(), "cores": os.cpu_count() or 0,
            "python": platform.python_version(), "have_numpy": HAVE_NUMPY}
    cpu = ""
    try:
        if platform.system() == "Linux":
            for ln in open("/proc/cpuinfo"):
                if "model name" in ln:
                    cpu = ln.split(":", 1)[1].strip()
                    break
        elif platform.system() == "Darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    info["cpu"] = cpu or platform.processor() or platform.machine()
    info["env"] = ("colab" if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ else
                   "github_ci" if "GITHUB_ACTIONS" in os.environ else
                   "modal" if "MODAL_TASK_ID" in os.environ else
                   "kaggle" if "KAGGLE_KERNEL_RUN_TYPE" in os.environ else "local")
    return info


def work_unit(level):
    if HAVE_NUMPY:
        a = np.random.rand(level, level)
        float((a @ a).sum())
    else:
        x = 1.0
        for _ in range(level):
            x = (x * 1.0000001 + 1.0) % 1e6


def shallow_look():
    os.getloadavg()                          # cheap in-process self-read (raises on Windows/iOS)


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


def measure(fn, arg, n, warm=3):
    for _ in range(warm):
        fn(arg) if arg is not None else fn()
    c0 = cpu_seconds()
    for _ in range(n):
        fn(arg) if arg is not None else fn()
    return (cpu_seconds() - c0) / n


def stats(xs):
    return {"mean": statistics.mean(xs), "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0}


def main():
    work_costs = {str(lvl): stats([measure(work_unit, lvl, 200) for _ in range(REPS)])
                  for lvl in WORK_LEVELS}
    look_counts = {"shallow_inproc": (shallow_look, 8000), "proc_inproc": (proc_look, 2000),
                   "deep_subprocess": (deep_look, 60)}
    modalities = {}
    for name, (fn, n) in look_counts.items():
        try:
            fn()
        except (OSError, ValueError, subprocess.SubprocessError, AttributeError, PermissionError) as e:
            modalities[name] = {"available": False, "reason": type(e).__name__}
            continue
        modalities[name] = {"available": True, **stats([measure(fn, None, n) for _ in range(REPS)])}
    result = {"machine": machine_id(), "reps": REPS, "work_levels": WORK_LEVELS,
              "work_costs_s": work_costs, "look_costs_s": modalities}
    out = json.dumps(result, indent=2)
    sys.stdout.write(out + "\n")
    with open("benchmark_result.json", "w") as f:
        f.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
