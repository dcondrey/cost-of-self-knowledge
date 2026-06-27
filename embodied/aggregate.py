"""Aggregate benchmark_result.json files from every machine into the cross-substrate table + figure.

Drop each machine's benchmark_result.json into results/ (named however you like). This reads them
all, builds the look_frac-per-modality-per-machine table (mean +/- std), and plots each machine
against the break-even band, showing the generality claim: on every real substrate, shallow
introspection is cheap (below break-even) and deep self-telemetry is dear (above it).
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

BREAKEVEN_LO, BREAKEVEN_HI = 0.04, 0.31      # from study.py main + sensitivity range
MODALITIES = ["shallow_inproc", "proc_inproc", "deep_subprocess"]


def load():
    rows = []
    for path in sorted(glob.glob("results/*.json")):
        d = json.load(open(path))
        m = d["machine"]
        label = f"{m.get('cpu') or m.get('system')} ({m.get('env','?')})"[:34]
        row = {"machine": label, "system": m.get("system"), "env": m.get("env"),
               "cores": m.get("cores"), "work_us": d.get("work_unit_s_mean", 0) * 1e6, "file": path}
        for k in MODALITIES:
            md = d["modalities"].get(k, {})
            row[k] = md.get("look_frac_mean") if md.get("available") else None
            row[k + "_std"] = md.get("look_frac_std", 0.0) if md.get("available") else None
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = load()
    if df.empty:
        log.info("no results/*.json found. Run benchmark.py on each machine and drop the JSON in results/.")
        return 1
    log.info(f"Aggregated {len(df)} machine(s).")
    show = ["machine", "system", "cores", "work_us"] + MODALITIES
    log.info(df[show].to_string(index=False, float_format=lambda x: f"{x:.4f}" if x == x else "n/a"))
    df.to_csv("benchmark_cross_machine.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(df) + 2))
    ax.axvspan(BREAKEVEN_LO, BREAKEVEN_HI, color="grey", alpha=0.18, label="break-even band (model)")
    colors = {"shallow_inproc": "green", "proc_inproc": "orange", "deep_subprocess": "red"}
    marks = {"shallow_inproc": "o", "proc_inproc": "s", "deep_subprocess": "D"}
    for k in MODALITIES:
        ys, xs, xe = [], [], []
        for i, r in df.iterrows():
            if r[k] is not None and r[k] == r[k]:
                ys.append(i); xs.append(r[k]); xe.append(r[k + "_std"] or 0)
        if xs:
            ax.errorbar(xs, ys, xerr=xe, fmt=marks[k], color=colors[k], ms=7, capsize=3,
                        label=k.replace("_", " "), ls="none")
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df["machine"], fontsize=8)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("cost of self-knowledge  (look_frac, log scale)")
    ax.set_title("The cost of self-introspection across real substrates")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig("fig3_cross_machine.png", dpi=150); plt.close(fig)

    deep = df["deep_subprocess"].dropna()
    shal = df["shallow_inproc"].dropna()
    log.info("-" * 60)
    if len(deep):
        log.info(f"deep self-telemetry: {deep.min():.3f}-{deep.max():.3f} across machines "
                 f"(all {'ABOVE' if deep.min() > BREAKEVEN_HI else 'mixed vs'} break-even)")
    if len(shal):
        log.info(f"shallow introspection: {shal.min():.4f}-{shal.max():.4f} "
                 f"(all {'BELOW' if shal.max() < BREAKEVEN_LO else 'mixed vs'} break-even)")
    log.info("Wrote benchmark_cross_machine.csv, fig3_cross_machine.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
