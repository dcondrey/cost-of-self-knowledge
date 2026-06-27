"""Irreversible cost-of-memory calibration.

Does writing data (the act of remembering / persisting state) measurably and PERMANENTLY age
the SSD? Writes modest incompressible data to the internal SSD, measures the increment in the
drive's lifetime bytes-written counter (irreversible) and any temperature rise (recoverable).
The temp file is deleted afterward, but the wear it caused is not - that is exactly the point:
remembering costs the body something it never gets back.
"""
import logging
import os
import re
import subprocess
import sys
import tempfile
import time

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("wear")

DEV = "/dev/disk0"            # internal SSD
UNIT_BYTES = 512_000         # NVMe "Data Units Written" = 1000 * 512 bytes per unit


def smart_ssd():
    """Return (temperature_C, data_units_written) for the internal SSD, or (None, None)."""
    try:
        out = subprocess.run(["smartctl", "-a", DEV], capture_output=True, text=True, timeout=10.0).stdout
    except (subprocess.SubprocessError, OSError):
        return None, None
    t = re.search(r"Temperature:\s+(\d+)\s+Celsius", out)
    w = re.search(r"Data Units Written:\s+([\d,]+)", out)
    return (int(t.group(1)) if t else None,
            int(w.group(1).replace(",", "")) if w else None)


def write_mb(mb):
    """Write `mb` MB of incompressible data to the internal SSD, fsync, delete. Return seconds."""
    chunk = os.urandom(8 * 1024 * 1024)          # incompressible so it truly reaches the drive
    fd, path = tempfile.mkstemp(prefix="bodywear_")
    t0 = time.time()
    try:
        for _ in range(mb // 8):
            os.write(fd, chunk)
        os.fsync(fd)
    finally:
        os.close(fd)
        dt = time.time() - t0
        os.unlink(path)
    return dt


def calibrate(levels=(0, 512, 1024, 2048)):
    log.info(f"target {DEV} (internal SSD)  -  remembering ages the body, irreversibly")
    log.info("=" * 70)
    base_t, base_w = smart_ssd()
    if base_w is None:
        log.info("could not read Data Units Written (need smartctl on the internal NVMe); abort")
        return 1
    log.info(f"{'write_MB':>9} {'GB_written_delta':>17} {'temp_C':>7} {'dT':>4} {'MB/s':>7}")
    prev_w, rows = base_w, []
    for mb in levels:
        dt = write_mb(mb) if mb else 0.0
        subprocess.run(["sync"], timeout=10.0)
        time.sleep(1.0)
        t, w = smart_ssd()
        d_gb = round((w - prev_w) * UNIT_BYTES / 1e9, 3) if w is not None else None
        dT = (t - base_t) if (t is not None and base_t is not None) else None
        thr = round(mb / dt, 1) if dt > 0 else 0
        rows.append((mb, d_gb, t, dT))
        log.info(f"{mb:>9} {str(d_gb):>17} {str(t):>7} {str(dT):>4} {thr:>7}")
        prev_w = w

    log.info("-" * 70)
    incr = [d for mb, d, t, dT in rows if mb > 0 and d and d > 0]
    total = round(sum(d for _, d, _, _ in rows if d), 2)
    if incr and total > 0:
        log.info(f"VERDICT: writing PERMANENTLY increments lifetime bytes-written - the "
                 f"cost-of-memory channel is real and measurable. This run cost ~{total} GB of "
                 f"unrecoverable wear (negligible vs the 184 TB lifetime, but genuinely permanent).")
        thermal = [dT for _, _, _, dT in rows if dT]
        log.info("Thermal response to brief writes is "
                 + ("present." if any(x and x > 0 for x in thermal) else "minimal (brief bursts "
                    "don't heat a fast SSD much; sustained writes or the sudo chip-temp sense needed)."))
        return 0
    log.info("VERDICT: bytes-written did not increment as expected; investigate write path / TMPDIR.")
    return 1


if __name__ == "__main__":
    sys.exit(calibrate())
