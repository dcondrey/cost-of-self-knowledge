"""Make-or-break: is self-measurement perturbation REDUCIBLE or IRREDUCIBLE?

A system reads its own temperature. Reading is computation; computation heats the substrate; the heat
changes the temperature read. We ramp self-measurement intensity UP then DOWN and measure:

  delta_min  : temperature rise from the MINIMUM possible self-observation. >0 => you can never
               sample your own unperturbed state.
  hysteresis : temp(intensity) up-ramp vs down-ramp. A nonzero loop => the reading depends on
               measurement HISTORY (thermal mass), so the true idle state is unrecoverable from any
               single reading => IRREDUCIBLE. ~Zero loop => memoryless removable offset => reducible.
  T0_recover : can we extrapolate intensity->0 to recover the idle temperature, and is that estimate
               stable between the up and down ramps? Unstable => irreducible.

Thermal source auto-detected: Linux /sys/class/thermal (perturb = CPU compute; clean, no wear) or
macOS SSD via smartctl (perturb = disk writes; costs a little wear, reported).
"""
import glob
import logging
import os
import platform
import statistics
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("selfmeasure")

HOLD_S = 25.0          # thermal settling per intensity level
SETTLE_READS = 5       # temp reads averaged at the end of each hold
LEVELS = [0, 1, 2, 3, 4]   # intensity levels (cores busy, or write-rate units)
MAX_WRITE_GB = 20.0    # hard wear cap for the macOS SSD path
_written_mb = [0.0]    # running wear tracker


def _linux_temp():
    vals = []
    for f in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(f) as fh:
                vals.append(int(fh.read().strip()) / 1000.0)
        except (OSError, ValueError):
            pass
    return max(vals) if vals else None


def _mac_ssd_temp():
    try:
        out = subprocess.run(["smartctl", "-a", "/dev/disk0"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for ln in out.splitlines():
        if "Temperature:" in ln and "Celsius" in ln:
            for tok in ln.split():
                if tok.isdigit():
                    return float(tok)
    return None


def detect_source():
    if platform.system() == "Linux" and _linux_temp() is not None:
        return "cpu", _linux_temp
    if platform.system() == "Darwin" and _mac_ssd_temp() is not None:
        return "ssd", _mac_ssd_temp
    return None, None


def _busy(stop_t):
    x = 1.0
    while time.time() < stop_t:
        for _ in range(20000):
            x = (x * 1.0000001 + 1.0) % 1e6


def perturb_compute(level, dur):
    import multiprocessing as mp
    if level <= 0:
        time.sleep(dur)
        return 0.0
    stop = time.time() + dur
    procs = [mp.Process(target=_busy, args=(stop,)) for _ in range(level)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return 0.0


def perturb_write(level, dur, rate_unit_mb_s=120):
    """level scales the sustained write rate (level*rate_unit MB/s) for dur seconds; rate-limited and
    capped at MAX_WRITE_GB total to bound SSD wear. Returns MB written this call."""
    if level <= 0:
        time.sleep(dur)
        return 0.0
    target = level * rate_unit_mb_s
    chunk = os.urandom(8 * 1024 * 1024)
    written = 0.0
    stop = time.time() + dur
    path = "/tmp/selfmeasure_blob"
    while time.time() < stop:
        if _written_mb[0] >= MAX_WRITE_GB * 1024:
            time.sleep(max(0.0, stop - time.time()))
            break
        t0 = time.time()
        with open(path, "wb") as f:
            f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        written += 8
        _written_mb[0] += 8
        ideal = 8.0 / target
        slack = ideal - (time.time() - t0)
        if slack > 0:
            time.sleep(slack)
    try:
        os.remove(path)
    except OSError:
        pass
    return written


def read_temp_avg(read_fn):
    vals = []
    for _ in range(SETTLE_READS):
        t = read_fn()
        if t is not None:
            vals.append(t)
        time.sleep(0.4)
    return statistics.mean(vals) if vals else None


def ramp(perturb, read_fn, levels, label):
    temps, mb = {}, 0.0
    for lv in levels:
        m = perturb(lv, HOLD_S)
        mb += m or 0.0
        t = read_temp_avg(read_fn)
        temps[lv] = t
        log.info(f"  [{label}] intensity={lv}  temp={t:.2f}")
    return temps, mb


def main():
    src, read_fn = detect_source()
    if not src:
        log.info("no readable, perturbable thermal source (need Linux /sys thermal or macOS smartctl). "
                 "On macOS CPU temp needs sudo; this uses SSD temp + writes instead.")
        return 1
    if src == "ssd" and not os.environ.get("SELFMEASURE_ALLOW_WEAR"):
        log.info(f"macOS has no no-sudo CPU thermal; the only source here is SSD temp, perturbed by "
                 f"writes (up to ~{MAX_WRITE_GB:.0f} GB of irreversible wear). To consent: "
                 f"SELFMEASURE_ALLOW_WEAR=1 python3 selfmeasure.py. Better: run on a Linux box "
                 f"(CPU compute + /sys thermal, zero wear).")
        return 2
    perturb = perturb_compute if src == "cpu" else perturb_write
    log.info(f"source={src}  perturb={'CPU compute' if src=='cpu' else 'SSD writes'}  "
             f"hold={HOLD_S}s  levels={LEVELS}")
    log.info(f"cooling to baseline...")
    perturb(0, HOLD_S)
    base = read_temp_avg(read_fn)
    log.info(f"baseline idle temp = {base:.2f}")

    up, mb_up = ramp(perturb, read_fn, LEVELS, "up")
    down, mb_down = ramp(perturb, read_fn, list(reversed(LEVELS)), "down")

    inner = [lv for lv in LEVELS if up.get(lv) is not None and down.get(lv) is not None]
    hyst = statistics.mean([down[lv] - up[lv] for lv in inner]) if inner else float("nan")
    delta_min = (up[LEVELS[1]] - base) if up.get(LEVELS[1]) is not None else float("nan")
    # T0 recovery: linear fit temp vs intensity, intercept; compare up vs down ramps
    def intercept(d):
        xs = [lv for lv in LEVELS if d.get(lv) is not None]
        ys = [d[lv] for lv in xs]
        n = len(xs)
        if n < 2:
            return float("nan")
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs) or 1e-9
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        return my - slope * mx
    t0_up, t0_down = intercept(up), intercept(down)

    log.info("=" * 64)
    log.info(f"baseline idle           : {base:.2f}")
    log.info(f"delta_min (cheapest look): {delta_min:+.2f}   (floor; >0 => can't sample unperturbed state)")
    log.info(f"HYSTERESIS (down - up)   : {hyst:+.2f}   (nonzero => history-dependent => IRREDUCIBLE)")
    log.info(f"T0 extrapolated up/down  : {t0_up:.2f} / {t0_down:.2f}   (gap {abs(t0_up-t0_down):.2f})")
    if src == "ssd":
        log.info(f"wear cost of this run    : ~{(mb_up+mb_down)/1024:.2f} GB written")
    irreducible = (not (hyst != hyst)) and abs(hyst) > 0.3 and abs(t0_up - t0_down) > 0.3
    if irreducible:
        log.info("=> IRREDUCIBLE: self-measurement leaves a history-dependent trace; the unperturbed "
                 "state is unrecoverable from any reading. Self-reference is load-bearing.")
    else:
        log.info("=> reducible / inconclusive: perturbation looks like a removable offset here "
                 "(or signal below thermal resolution). Honest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
