"""A small, tightly-bounded body that can really die - two ways, in real opposition - and the
honest question of whether knowing yourself is worth its cost.

Grounded in the real memory hierarchy (no invented drives):
  - WORK fills RAM (real bytearrays). Holding more RAM makes you more productive...
  - ...but RAM is capped: exceed it and you die of OOM (a RECOVERABLE cliff).
  - To free RAM you must CONSOLIDATE: write to the SSD - which spends irreversible WEAR.
  - Exhaust your wear budget and you die WORN OUT (an IRREVERSIBLE cliff).
Escaping one death causes the other. A blind agent acts on a fixed schedule. A self-regulating agent
MONITORS its real RAM state and consolidates just-in-time - but looking costs (the observer effect:
checking yourself takes compute that could have been work). We sweep that look-cost: does knowing
yourself still pay when knowing yourself isn't free, and where is the break-even? Accelerated so a
whole life plays out in a few hundred ticks.
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("life")

MEM_CAP_MB = 300.0
WEAR_BUDGET = 1500.0
MEM_PER_WORK = 4.0
WEAR_PER_MB = 0.6
MAX_TICKS = 5000


def live(decide, look_frac):
    held = []
    mem_mb, wear_left, ach = 0.0, WEAR_BUDGET, 0.0
    for t in range(MAX_TICKS):
        consolidate, looked = decide(mem_mb, wear_left)
        if consolidate and mem_mb > 0:
            wear_left -= mem_mb * WEAR_PER_MB
            held.clear()
            mem_mb = 0.0
            if wear_left <= 0:
                held.clear()
                return t + 1, round(ach, 0), "worn out"
        productivity = 1.0 + mem_mb / 100.0
        if looked:
            productivity *= (1.0 - look_frac)        # looking ate part of this tick's compute
        ach += productivity
        held.append(bytearray(int(MEM_PER_WORK * 1024 * 1024)))
        mem_mb += MEM_PER_WORK
        if mem_mb > MEM_CAP_MB:
            held.clear()
            return t + 1, round(ach, 0), "OOM"
    held.clear()
    return MAX_TICKS, round(ach, 0), "survived"


STRATEGIES = {
    "hoarder": lambda m, w: (False, False),
    "scribe": lambda m, w: (True, False),
    "periodic30": (lambda: (lambda m, w, s=[0]: (s.__setitem__(0, s[0] + 1), (s[0] % 30 == 0, False))[1]))(),
    "adaptive(look)": lambda m, w: (m > 0.8 * MEM_CAP_MB, True),
}


def main():
    log.info(f"Tight body: RAM cap {MEM_CAP_MB:.0f}MB (OOM) vs wear {WEAR_BUDGET:.0f} (mortality). "
             f"Self-regulation must pay a look-cost; blind strategies don't look.")
    log.info("=" * 78)
    log.info(f"  achievement by strategy, across the cost of self-knowledge (look_frac):")
    log.info(f"{'look_cost':>10} " + "".join(f"{k:>14}" for k in STRATEGIES))
    breakeven = None
    for lf in [0.0, 0.05, 0.10, 0.20, 0.40]:
        achs = {k: live(d, lf)[1] for k, d in STRATEGIES.items()}
        blind_best = max(v for k, v in achs.items() if k != "adaptive(look)")
        win = achs["adaptive(look)"] > blind_best
        if breakeven is None and not win:
            breakeven = lf
        mark = " <-adaptive wins" if win else " <-blind wins"
        log.info(f"{lf:>10.2f} " + "".join(f"{achs[k]:>14.0f}" for k in STRATEGIES) + mark)
    log.info("-" * 78)
    if breakeven is None:
        log.info("VERDICT: self-regulation pays at EVERY look-cost tested - knowing yourself is worth "
                 "it across the board when the body can really die two opposing ways.")
    elif breakeven == 0.0:
        log.info("VERDICT: self-regulation never pays once looking costs anything - consistent with the "
                 "earlier grounded null. Knowing yourself is not worth its price here (honest).")
    else:
        log.info(f"VERDICT: self-regulation pays UNTIL look-cost ~{breakeven:.2f}, then blind scheduling "
                 f"wins. There is a real break-even price of self-knowledge - below it, monitoring the "
                 f"body that can die is worth the compute it costs; above it, willed blindness is optimal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
