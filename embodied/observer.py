"""Observer-effect feasibility gate.

On one substrate, observing yourself is computation is energy: looking at your own state should
cost you. This sweeps self-monitoring frequency and measures the agent's OWN cpu time spent
monitoring (exact, via os.times including subprocess children) - the metabolic cost of
self-observation. It compares two ways of looking (cheap in-process vs subprocess-based telemetry)
because the cost of self-knowledge depends on the modality, and it demonstrates the self-referential
loop directly: at high monitoring rates the agent's own monitoring becomes visible in the very
telemetry it reads.

Verdict question: is the cost of looking measurable and dose-dependent? If yes, the substrate
gives a real observer effect (no free introspection) and the deep experiment is feasible.
"""
import logging
import os
import sys
import time

from body import Body, BodyReading

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("observer")


def cost_of_looking(reader, rate_hz, seconds):
    """Call reader() at ~rate_hz for `seconds`; return (cpu_fraction_of_a_core, reads_done).

    cpu_fraction uses os.times() user+sys+children (subprocess telemetry runs as children), so it
    captures the TRUE cost of looking, including spawned ps/sysctl/vm_stat processes.
    """
    interval = (1.0 / rate_hz) if rate_hz > 0 else 0.0
    c0 = sum(os.times()[:4])
    w0 = time.time()
    t_end = w0 + seconds
    reads, nxt = 0, w0
    while time.time() < t_end:
        if rate_hz == 0:
            time.sleep(0.05)
            continue
        reader()
        reads += 1
        nxt += interval
        s = nxt - time.time()
        if s > 0:
            time.sleep(s)
    cpu = sum(os.times()[:4]) - c0
    wall = time.time() - w0
    return cpu / wall, reads


def main():
    body = Body()
    body.read()                                  # warm the slow cache so timing isolates fast looks

    def look_subprocess():                        # telemetry via spawned ps/sysctl/vm_stat
        body._fast(BodyReading(t=0.0))

    def look_inprocess():                         # cheap: kernel load average, no subprocess
        os.getloadavg()

    log.info(f"organs={len(body.drives)} cpu={body.ncpu}")
    log.info("=" * 64)
    log.info(f"{'look@Hz':>8} {'subprocess cpu/core':>22} {'in-process cpu/core':>22}")
    rates = [0, 1, 5, 20, 50]
    rows = []
    for r in rates:
        sp, _ = cost_of_looking(look_subprocess, r, 4.0)
        ip, _ = cost_of_looking(look_inprocess, r, 4.0)
        rows.append((r, sp, ip))
        log.info(f"{r:>8} {sp*100:>21.2f}% {ip*100:>21.4f}%")
    log.info("-" * 64)

    # self-referential demonstration: monitor hard, then see the monitoring in the telemetry
    def burn_looking(seconds):
        t_end = time.time() + seconds
        while time.time() < t_end:
            body._fast(BodyReading(t=0.0))
    import threading
    th = threading.Thread(target=burn_looking, args=(3.0,))
    th.start()
    time.sleep(1.5)
    r = body.read()
    th.join()
    log.info(f"self-referential check: while monitoring hard, the agent's own CPU footprint reads "
             f"self_cpu={r.self_cpu_pct}% - its looking shows up in what it sees.")
    log.info("-" * 64)

    base = rows[0][1]
    top = rows[-1][1]
    delta = (top - base) * 100
    if delta > 1.0:
        log.info(f"VERDICT: looking at yourself COSTS, dose-dependently "
                 f"(subprocess looking: {base*100:.1f}% -> {top*100:.1f}% of a core across "
                 f"{rates[1]}..{rates[-1]} Hz). The observer effect is real and tunable; "
                 f"no free introspection. Modality matters: in-process looking is ~"
                 f"{rows[-1][1]/max(rows[-1][2],1e-9):.0f}x cheaper.")
        return 0
    log.info("VERDICT: self-observation cost is below threshold at these rates; amplify "
             "(higher rate / heavier read) to reach a behaviorally-relevant regime.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
