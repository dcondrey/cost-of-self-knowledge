"""Self-contained phone/constrained-device build of the self-introspection benchmark.

Standalone (no imports from benchmark.py) so it runs from a single downloaded file on a-shell / iOS
/ any minimal Python. Robust by design: every modality probe catches ALL exceptions and records why
(constrained platforms raise platform-specific errors), and the whole run is wrapped so any fatal
error is reported as ONE line (+ uploaded) instead of a wall of tracebacks. Prints a REPORT-BACK
block at the very bottom: a short paste URL if upload works, and always a compact one-line summary.
"""
import json
import os
import platform
import statistics
import sys
import traceback

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

REPS = 9
WORK_LEVELS = [64, 128, 256, 512] if HAVE_NUMPY else [50_000, 100_000, 200_000, 400_000]


def out(*a):
    sys.stdout.write(" ".join(str(x) for x in a) + "\n")


def cpu_seconds():
    return sum(os.times()[:4])


def machine_id():
    return {"platform": platform.platform(), "machine": platform.machine(),
            "system": platform.system(), "cores": os.cpu_count() or 0,
            "python": platform.python_version(), "have_numpy": HAVE_NUMPY,
            "cpu": platform.processor() or platform.machine(), "env": "phone"}


def work_unit(level):
    if HAVE_NUMPY:
        a = np.random.rand(level, level)
        float((a @ a).sum())
    else:
        x = 1.0
        for _ in range(level):
            x = (x * 1.0000001 + 1.0) % 1e6


def shallow_look():
    os.getloadavg()


def proc_look():
    for p in ("/proc/stat", "/proc/meminfo"):
        with open(p) as f:
            f.read()


def deep_look():
    import subprocess
    s = platform.system()
    cmd = (["sh", "-c", "cat /proc/stat /proc/meminfo 2>/dev/null"] if s == "Linux" else
           ["sh", "-c", "vm_stat; ps -A -o %cpu="] if s in ("Darwin", "iOS") else ["uptime"])
    subprocess.run(cmd, capture_output=True, timeout=15)


def measure(fn, arg, n, warm=3):
    for _ in range(warm):
        fn(arg) if arg is not None else fn()
    c0 = cpu_seconds()
    for _ in range(n):
        fn(arg) if arg is not None else fn()
    return (cpu_seconds() - c0) / n


def stats(xs):
    return {"mean": statistics.mean(xs), "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0}


def build_result():
    work_costs = {str(lvl): stats([measure(work_unit, lvl, 200) for _ in range(REPS)])
                  for lvl in WORK_LEVELS}
    modalities = {}
    for name, fn, n in (("shallow_inproc", shallow_look, 8000), ("proc_inproc", proc_look, 2000),
                        ("deep_subprocess", deep_look, 60)):
        try:
            fn()
        except BaseException as e:                       # any platform-specific failure -> skip, record why
            modalities[name] = {"available": False, "reason": type(e).__name__}
            continue
        try:
            modalities[name] = {"available": True, **stats([measure(fn, None, n) for _ in range(REPS)])}
        except BaseException as e:
            modalities[name] = {"available": False, "reason": "measure:" + type(e).__name__}
    return {"machine": machine_id(), "reps": REPS, "work_levels": WORK_LEVELS,
            "work_costs_s": work_costs, "look_costs_s": modalities}


def post(text):
    try:
        import urllib.request
        req = urllib.request.Request("https://paste.rs/", data=text.encode(), method="POST")
        return urllib.request.urlopen(req, timeout=20).read().decode().strip()
    except BaseException:
        return None


def report(result):
    def us(x):
        return "NA" if x is None else format(x * 1e6, ".2f")

    def lk(name):
        m = result["look_costs_s"].get(name, {})
        return us(m["mean"]) if m.get("available") else "NA"
    wm = ",".join(us(result["work_costs_s"][str(l)]["mean"]) for l in WORK_LEVELS)
    line = ("SELFus cores=%s work=%s shallow=%s proc=%s deep=%s"
            % (result["machine"]["cores"], wm, lk("shallow_inproc"), lk("proc_inproc"), lk("deep_subprocess")))
    url = post(json.dumps(result))
    out("\n\n========= REPORT BACK (scroll here) =========")
    if url:
        out("URL:", url)
    out(line)
    out("=============================================")


def main():
    try:
        report(build_result())
    except BaseException:
        tb = traceback.format_exc()
        url = post("PHONE ERROR\n" + tb)
        out("\n\n========= ERROR - REPORT BACK =========")
        if url:
            out("URL:", url)
        out("LAST:", tb.strip().splitlines()[-1])
        out("=======================================")


main()
