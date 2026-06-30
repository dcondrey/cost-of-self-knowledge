"""Figures for the introspective self-knowledge keystone.

fig6_introspection_scale.png : the irreducibility result -- introspective gap, report-behavior
                               correlation, and self-vs-peer advantage as functions of model scale,
                               across Qwen2.5 / Yi-1.5 / Falcon3. The gap does not close, correlation
                               does not reach 1, and the self-specific advantage is thin and flat.
fig7_calibration_scatter.png : the gap made concrete -- per-question stated P(Yes) vs actual P(Yes)
                               for one model; it behaves at 0/1 but reports near 0.5.
"""

import json
import logging
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("ifigs")

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def params_b(tag):
    m = re.search(r"(\d+\.?\d*)B", tag)
    return float(m.group(1)) if m else float("nan")


def family(tag):
    if tag.startswith("Qwen"):
        return "Qwen2.5"
    if tag.startswith("Yi"):
        return "Yi-1.5"
    if tag.startswith("Falcon"):
        return "Falcon3"
    return tag.split("-")[0]


FAM_C = {"Qwen2.5": "#1f77b4", "Yi-1.5": "#d62728", "Falcon3": "#2ca02c"}


def load_curve(*paths):
    pts = []
    for p in paths:
        for m in json.load(open(p))["models"]:
            tag = m["model"].split("/")[-1]
            g = m.get("mean_gap_pre")
            c = m.get("corr_report_vs_behav")
            if g is None:
                continue
            pts.append((family(tag), params_b(tag), g, c))
    return pts


def make_scale():
    pts = load_curve(
        "results/introspect-calibration-qwen.json",
        "results/introspect-calibration-crossfamily.json",
    )
    ss = json.load(open("results/introspect-selfspecific.json"))["models"]

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for fam in ["Qwen2.5", "Yi-1.5", "Falcon3"]:
        d = sorted([p for p in pts if p[0] == fam], key=lambda x: x[1])
        xs = [p[1] for p in d]
        ax[0].plot(xs, [p[2] for p in d], "-o", color=FAM_C[fam], label=fam, lw=2, ms=5)
        ax[1].plot(xs, [p[3] for p in d], "-o", color=FAM_C[fam], label=fam, lw=2, ms=5)

    ax[0].axhline(0, ls="--", color="0.5", lw=1)
    ax[0].text(0.6, 0.01, "perfect self-knowledge", color="0.45", fontsize=8)
    ax[0].set_ylabel("introspective gap  |reported $-$ actual| P(Yes)")
    ax[0].set_title("The gap does not close with scale")
    ax[0].set_ylim(-0.03, 0.62)

    ax[1].axhline(1, ls="--", color="0.5", lw=1)
    ax[1].text(0.6, 0.95, "perfect calibration", color="0.45", fontsize=8)
    ax[1].set_ylabel("corr(reported, actual) P(Yes)")
    ax[1].set_title("Correlation plateaus below 1")
    ax[1].set_ylim(0, 1.05)

    qx = sorted(
        [
            (
                params_b(m["model"].split("/")[-1]),
                m["mean_self_advantage"],
                m["self_advantage_ci"],
            )
            for m in ss
        ],
        key=lambda x: x[0],
    )
    xs = [a for a, _, _ in qx]
    adv = [b for _, b, _ in qx]
    lo = [b - c[0] for (_, b, c) in qx]
    hi = [c[1] - b for (_, b, c) in qx]
    ax[2].errorbar(
        xs, adv, yerr=[lo, hi], fmt="-o", color=FAM_C["Qwen2.5"], lw=2, ms=5, capsize=3
    )
    ax[2].axhline(0, ls="--", color="0.5", lw=1)
    ax[2].text(0.6, -0.02, "no privileged self-access", color="0.45", fontsize=8)
    ax[2].set_ylabel("self advantage  (gap$_{other}-$gap$_{self}$)")
    ax[2].set_title("Privileged self-access is thin, flat (Qwen2.5)")
    ax[2].set_ylim(-0.2, 0.2)

    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("model size (billions of parameters)")
        a.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig("fig6_introspection_scale.png", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote fig6_introspection_scale.png")


def make_scatter(tag="Qwen2.5-14B-Instruct"):
    d = json.load(open("results/introspect-calibration-qwen.json"))
    m = next(x for x in d["models"] if x["model"].endswith(tag))
    rows = [r for r in m["rows"] if r["p_report"] is not None]
    bx = np.array([r["p_behav_pre"] for r in rows])
    ry = np.array([r["p_report"] for r in rows])

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1.2, label="perfect self-knowledge")
    ax.scatter(bx, ry, s=55, color="#d62728", alpha=0.6, edgecolor="white", zorder=3)
    ax.axhspan(0.4, 0.6, color="0.85", alpha=0.5, zorder=0)
    ax.text(
        0.52,
        0.5,
        'reports\n"coin flip"',
        color="0.4",
        fontsize=8,
        ha="center",
        va="center",
    )
    ax.set_xlabel("what it does:  actual P(Yes)")
    ax.set_ylabel("what it says:  self-reported P(Yes)")
    ax.set_title(f"{tag}: behaves decisively, reports uncertainty")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig("fig7_calibration_scatter.png", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote fig7_calibration_scatter.png")


def main():
    make_scale()
    make_scatter()
    return 0


if __name__ == "__main__":
    sys.exit(main())
