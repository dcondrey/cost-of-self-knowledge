"""The agent's muscles: controllable compute that produces real, sensible bodily effects.

The complement to body.py (senses). burn(cores, seconds) spins real CPU work; the calibration
sweeps effort level and measures whether the body's signals respond ABOVE the idle noise floor.
This is the feasibility gate: if the agent's own thinking cannot measurably move its own body,
the felt-cost-of-cognition idea is not viable on this hardware.

Primary effort signal is total process CPU% (fast, instantaneous). Load average and drive
temps are also recorded but respond slowly (long time constants), so they are secondary here.
"""
import logging
import multiprocessing as mp
import re
import subprocess
import sys
import time

from body import Body

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("effort")


def _spin(stop_t):
    x = 1.0
    while time.time() < stop_t:
        for _ in range(50000):
            x = (x * 1.0000001 + 1.0) % 1e6


def burn_async(cores, seconds):
    """Start `cores` busy worker processes for `seconds`; return the process list (not joined)."""
    stop_t = time.time() + seconds
    procs = [mp.Process(target=_spin, args=(stop_t,)) for _ in range(cores)]
    for p in procs:
        p.start()
    return procs


def total_cpu_pct():
    """Sum of all processes' recent CPU% (fast, instantaneous-ish). ~10 cores -> 1000% max."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "%cpu="], capture_output=True, text=True, timeout=3.0).stdout
        return round(sum(float(v) for v in out.split()), 1)
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return round(m, 1), round(sd, 1)


def calibrate(levels=(0, 2, 4, 8), burn_s=8.0):
    body = Body()
    log.info(f"organs: {', '.join(body.drives) or 'none'}  cpu={body.ncpu}")
    log.info("=" * 70)

    base = [total_cpu_pct() for _ in range(5) if not time.sleep(0.4)]
    base_m, base_sd = _stats(base)
    log.info(f"idle noise floor: total_cpu = {base_m}% +- {base_sd}%")
    log.info("-" * 70)
    log.info(f"{'cores':>6} {'total_cpu%':>12} {'delta':>9} {'detectable?':>12} {'load1':>7} {'ssd_C':>6}")

    rows = []
    for cores in levels:
        procs = burn_async(cores, burn_s) if cores else []
        time.sleep(3.0)                                   # let it ramp
        cpu = _stats([total_cpu_pct() for _ in range(4) if not time.sleep(0.3)])[0]
        r = body.read()
        for p in procs:
            p.join()
        delta = round((cpu or 0) - (base_m or 0), 1)
        detectable = delta > 3 * (base_sd or 1)
        ssd = next((v for k, v in r.temps_c.items() if "disk0" in k), None)
        rows.append((cores, cpu, delta, detectable))
        log.info(f"{cores:>6} {cpu:>12} {delta:>9} {str(detectable):>12} {r.load1:>7} {str(ssd):>6}")

    log.info("-" * 70)
    moved = [d for c, cpu, d, det in rows if c > 0 and det]
    if moved:
        per_core = [d / c for c, cpu, d, det in rows if c > 0 and det]
        log.info(f"VERDICT: agent's compute MOVES its body, detectably above noise "
                 f"(~{round(sum(per_core)/len(per_core))}% cpu per core). "
                 f"Recoverable-effort channel is viable.")
        return 0
    log.info("VERDICT: agent's compute did NOT move the body above noise. Feasibility problem.")
    return 1


if __name__ == "__main__":
    sys.exit(calibrate())
