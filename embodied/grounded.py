"""Grounded test: does self-monitoring have REAL instrumental value on the real machine?

No invented rewards. Real work = real compute (matmuls completed = throughput). Real stake =
real CPU contention from a fluctuating background load (the shared-substrate noise we measured).
Real cost of looking = the observer effect (sampling telemetry burns real time/CPU). The question:
does an agent that monitors real contention and times its work beat a blind agent that just works -
net of the real cost of monitoring? If yes, the "drive to check" is grounded in real physics, not
hand-coded. If no, on this hardware the real stakes are too mild for it to matter (an honest finding).
"""
import logging
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from body import Body
from effort import burn_async

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("grounded")

WORK_N = 320            # matmul size; one work unit = one NxN matmul
DURATION = 20.0         # seconds per policy
LOAD_THRESH = 6.0       # back off when (load1) above this


def background_contention(stop_t):
    """Fluctuating real CPU load: bursts of heavy compute on/off (a noisy neighbour)."""
    rng = np.random.default_rng(os.getpid())
    while time.time() < stop_t:
        cores = int(rng.integers(0, 9))          # 0..8 cores of real burn
        secs = float(rng.uniform(1.0, 3.0))
        procs = burn_async(cores, secs) if cores else []
        time.sleep(secs)
        for p in procs:
            p.join()


def work_unit():
    a = np.random.rand(WORK_N, WORK_N)
    (a @ a).sum()


def run_blind(duration):
    t_end = time.time() + duration
    units = 0
    while time.time() < t_end:
        work_unit()
        units += 1
    return units, 0


def run_adaptive(duration, body):
    t_end = time.time() + duration
    units, looks = 0, 0
    while time.time() < t_end:
        r = body.read()                          # REAL cost of looking (observer effect)
        looks += 1
        if r.load1 is None or r.load1 < LOAD_THRESH:
            work_unit()                          # work during low-contention windows
            units += 1
        else:
            time.sleep(0.05)                     # back off under real contention
    return units, looks


def main():
    body = Body()
    log.info("Grounded: real value of self-monitoring under real contention (no invented reward)")
    log.info("=" * 72)
    results = {}
    for name, fn in (("blind   (never look)", run_blind),
                     ("adaptive (look, time work to low load)", run_adaptive)):
        rng = np.random.default_rng(1)
        stop = time.time() + DURATION + 1.0
        bg = mp.Process(target=background_contention, args=(stop,))
        bg.start()
        time.sleep(1.0)                          # let contention ramp
        t0 = time.time()
        units, looks = (fn(DURATION) if fn is run_blind else fn(DURATION, body))
        wall = time.time() - t0
        bg.join()
        peak = body.read().load1
        results[name] = (units, looks, wall, peak)
        log.info(f"  {name:42s} work={units:5d}  looks={looks:5d}  ({units/wall:.0f} units/s)  peak_load={peak}")
    log.info("-" * 72)
    b = results["blind   (never look)"][0]
    a = results["adaptive (look, time work to low load)"][0]
    gain = (a - b) / max(b, 1) * 100
    log.info(f"VERDICT: monitoring changed real throughput by {gain:+.0f}% "
             f"(adaptive {a} vs blind {b} work units, net of real look cost).")
    if gain > 5:
        log.info("  => self-monitoring has REAL instrumental value here: the drive to check is "
                 "grounded in real contention physics, not invented.")
    elif gain < -5:
        log.info("  => monitoring COST more real throughput than it saved: on this mild machine, "
                 "checking does not pay - the dramatic behaviors needed invented stakes (honest).")
    else:
        log.info("  => roughly neutral: real stakes on this hardware are too mild for self-"
                 "monitoring to matter much (honest null; needs genuinely viability-constrained HW).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
