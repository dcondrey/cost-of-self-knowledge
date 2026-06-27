"""This machine's real, measured mortality - and the living-now vs living-long tradeoff.

No invented parameters: everything is read from the SSD's own wear telemetry (smartctl). The drive
has a finite write endurance; every write (every act of remembering / persisting state) permanently
spends a piece of it. So a continuously-running agent's CHOICE of how much to remember directly sets
how long it lives. This computes that real tradeoff from the machine's actual current state.

This is the grounded version of the cost-of-memory: not a reward we invented, but real irreversible
wear measured on real hardware. (Mechanically it is SSD endurance budgeting; the framing - an agent
pricing its own real, measured mortality into how much it dares to remember - is being-towards-death
on a real substrate, which the literature check found unclaimed.)
"""
import logging
import re
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("mortality")

DEV = "/dev/disk0"
UNIT_BYTES = 512_000


def read_wear():
    out = subprocess.run(["smartctl", "-a", DEV], capture_output=True, text=True, timeout=10.0).stdout
    duw = re.search(r"Data Units Written:\s+([\d,]+)", out)
    pct = re.search(r"Percentage Used:\s+(\d+)%", out)
    poh = re.search(r"Power On Hours:\s+([\d,]+)", out)
    if not (duw and pct and poh):
        return None
    tb_written = int(duw.group(1).replace(",", "")) * UNIT_BYTES / 1e12
    return tb_written, int(pct.group(1)), int(poh.group(1).replace(",", ""))


def main():
    w = read_wear()
    if not w:
        log.info("could not read SSD wear telemetry; need smartctl on the internal NVMe.")
        return 1
    tb_written, pct_used, poh = w
    if pct_used == 0:
        log.info(f"written {tb_written:.0f} TB but 0% used reported - endurance unknown on this drive.")
        return 1

    total = tb_written / (pct_used / 100.0)
    remaining = total - tb_written
    rate_tb_per_hr = tb_written / poh
    years = lambda hrs: hrs / (24 * 365.25)

    log.info("This machine's measured mortality (internal SSD)")
    log.info("=" * 62)
    log.info(f"  lifetime written      : {tb_written:8.1f} TB")
    log.info(f"  life used             : {pct_used:8d} %")
    log.info(f"  total write endurance : {total:8.0f} TB  (= written / %used)")
    log.info(f"  endurance REMAINING   : {remaining:8.0f} TB  (irreversible; never comes back)")
    log.info(f"  age                   : {poh:8d} h  ({years(poh):.2f} yr powered on)")
    log.info(f"  historical write rate : {rate_tb_per_hr*1000:8.1f} GB/h")
    log.info("-" * 62)
    log.info("  remembering rate vs remaining LIFESPAN (it writes itself to death):")
    log.info(f"  {'rate':>10} {'GB/h':>8} {'remaining life':>16}")
    for mult, label in [(0.25, "frugal"), (1.0, "as-now"), (2.0, "eager"), (5.0, "voracious")]:
        rate = rate_tb_per_hr * mult
        life_h = remaining / rate if rate > 0 else float("inf")
        log.info(f"  {label:>10} {rate*1000:8.1f} {years(life_h):13.1f} yr")
    log.info("=" * 62)
    base_years = years(remaining / rate_tb_per_hr)
    log.info(f"At its current pace this body has ~{base_years:.1f} years of writing left. An agent that "
             f"remembers more dies sooner; one that remembers less lives longer. That is the real, "
             f"measured stake - the choice of how much to commit to permanent memory IS a choice about "
             f"how long to live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
