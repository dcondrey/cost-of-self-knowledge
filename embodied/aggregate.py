"""Aggregate v2 benchmark_result.json files into the cross-substrate raw-cost table + figures.

Each machine's JSON (schema v2) reports raw CPU-second costs: cost-per-look per modality and
cost-per-work across a sweep of work-unit sizes. The substrate-invariant facts are the raw costs;
look_frac = cost_look / (cost_look + cost_work) is derived and work-scale-dependent. We therefore:
  - table  : raw cost-per-look per modality (denominator-free) + the work-cost range, per machine
  - W_crit : the critical work-unit cost (seconds) at which deep introspection reaches break-even
  - fig    : look_frac(deep) vs work-unit cost per machine, with the break-even band, showing the
             conclusion is robust across the whole work-scale sweep rather than one chosen unit
"""
import glob
import json
import logging
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("aggregate")

BE_LO, BE_HI = 0.04, 0.31                    # break-even band from study.py (main + sensitivity)
BE_MID = 0.175
MODS = ["shallow_inproc", "proc_inproc", "deep_subprocess"]


def load():
    machines = []
    for path in sorted(glob.glob("results/*.json")):
        d = json.load(open(path))
        if "work_costs_s" not in d:
            log.info(f"  skip {path} (old schema v1; re-run benchmark.py v2)")
            continue
        m = d["machine"]
        d["_label"] = f"{(m.get('cpu') or m.get('system'))[:26]} [{m.get('env','?')}]"
        d["_file"] = path
        machines.append(d)
    return machines


def look_cost(d, mod):
    md = d["look_costs_s"].get(mod, {})
    return md.get("mean") if md.get("available") else None


def work_curve(d):
    wc = d["work_costs_s"]
    return [(int(k), wc[k]["mean"]) for k in sorted(wc, key=int)]


def main():
    ms = load()
    if not ms:
        log.info("no v2 results/*.json found. Run benchmark.py (v2) on each machine.")
        return 1
    log.info(f"Aggregated {len(ms)} machine(s) (schema v2).")

    rows = []
    for d in ms:
        deep = look_cost(d, "deep_subprocess")
        w_crit = deep * (1 - BE_MID) / BE_MID if deep else None     # work cost where look_frac = break-even
        wmin = min(c for _, c in work_curve(d))
        wmax = max(c for _, c in work_curve(d))
        rows.append({"machine": d["_label"], "system": d["machine"].get("system"),
                     "shallow_s": look_cost(d, "shallow_inproc"),
                     "proc_s": look_cost(d, "proc_inproc"),
                     "deep_s": deep,
                     "work_s_range": f"{wmin:.2e}-{wmax:.2e}",
                     "Wcrit_deep_s": w_crit})
    df = pd.DataFrame(rows)
    log.info(df.to_string(index=False))
    df.to_csv("benchmark_cross_machine.csv", index=False)

    # Figure: look_frac(deep) vs work-unit cost per machine, with break-even band.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhspan(BE_LO, BE_HI, color="grey", alpha=0.18, label="break-even band (model)")
    for d in ms:
        deep = look_cost(d, "deep_subprocess")
        if not deep:
            continue
        xs = [c for _, c in work_curve(d)]
        ys = [deep / (deep + c) for c in xs]
        ax.plot(xs, ys, "-o", ms=4, label=d["_label"], alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("cost of one work-unit (CPU seconds, log)")
    ax.set_ylabel("look_frac of DEEP self-telemetry")
    ax.set_ylim(0, 1.02)
    ax.set_title("Cost of deep self-knowledge vs work scale, across substrates")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("fig3_cross_machine.png", dpi=150); plt.close(fig)

    sh = [r["shallow_s"] for r in rows if r["shallow_s"]]
    dp = [r["deep_s"] for r in rows if r["deep_s"]]
    wc = [r["Wcrit_deep_s"] for r in rows if r["Wcrit_deep_s"]]
    log.info("-" * 70)
    if sh:
        log.info(f"cost-per-look shallow : {min(sh):.2e}-{max(sh):.2e} s  (in-process, ~free)")
    if dp:
        log.info(f"cost-per-look deep    : {min(dp):.2e}-{max(dp):.2e} s  ({max(dp)/min(dp):.0f}x spread)")
    if wc:
        log.info(f"critical work scale   : {min(wc):.2e}-{max(wc):.2e} s  -- below this work-unit cost, "
                 f"deep self-knowledge is not worth its price")
    log.info("Wrote benchmark_cross_machine.csv, fig3_cross_machine.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
