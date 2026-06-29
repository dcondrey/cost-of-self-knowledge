"""Generate and polish the paper figures.

fig5_hysteresis.png : the standout -- thermal hysteresis loops (CPU + GPU) showing self-measurement
                      leaves a history-dependent trace, so the unperturbed state is unrecoverable.
fig1_breakeven.png  : polished break-even curve (self-regulation vs blind) with the measured machine
                      look-costs marked.
fig2_sensitivity.png: polished robustness heatmap of the break-even across body parameters.
"""

import json
import logging
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("figs")

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

UP, DN = "#1f77b4", "#d62728"


def hysteresis_panel(ax, path, title):
    d = json.load(open(path))
    lv = [int(x) for x in d["levels"]]
    up = [d["up"][str(l)] for l in lv]
    down = [d["down"][str(l)] for l in lv]
    ax.plot(lv, up, "-o", color=UP, lw=2.2, label="ramping up")
    ax.plot(lv, down, "-s", color=DN, lw=2.2, label="ramping down")
    ax.fill_between(lv, up, down, alpha=0.16, color=DN)
    ax.axhline(d["baseline"], ls="--", color="0.5", lw=1.2)
    ax.text(lv[0], d["baseline"], " true idle", color="0.4", va="bottom", fontsize=9)
    ax.annotate(
        "",
        xy=(0.15, down[0]),
        xytext=(0.15, up[0]),
        arrowprops=dict(arrowstyle="<->", color=DN, lw=1.4),
    )
    ax.text(
        0.3,
        (up[0] + down[0]) / 2,
        f"can't return\nto idle\n(+{down[0] - d['baseline']:.0f} C)",
        color=DN,
        fontsize=8.5,
        va="center",
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("self-measurement intensity")
    ax.set_ylabel("temperature (C)")
    ax.set_xticks(lv)
    ax.legend(fontsize=9, loc="lower right", frameon=False)


def make_hysteresis():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    hysteresis_panel(
        axs[0],
        "results/selfmeasure-linux-selfread.json",
        "CPU, self-telemetry read (airtight)",
    )
    hysteresis_panel(axs[1], "results/selfmeasure-gpu-colab.json", "GPU (Tesla T4)")
    fig.suptitle(
        "A system cannot read its own resting state: self-measurement leaves a "
        "history-dependent thermal trace",
        fontsize=12.5,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig("fig5_hysteresis.png", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote fig5_hysteresis.png")


def make_breakeven():
    df = pd.read_csv("study_main.csv")
    g = df["gap"].values
    lf = df["look_frac"].values
    be = next(
        (
            lf[i] + (lf[i + 1] - lf[i]) * g[i] / (g[i] - g[i + 1])
            for i in range(len(g) - 1)
            if g[i] >= 0 >= g[i + 1]
        ),
        float("nan"),
    )
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(
        df.look_frac,
        df.adaptive,
        "-o",
        ms=4,
        color="#2ca02c",
        lw=2,
        label="self-regulating (monitors, pays look-cost)",
    )
    ax.fill_between(
        df.look_frac,
        df.adaptive - df.adaptive_ci,
        df.adaptive + df.adaptive_ci,
        alpha=0.18,
        color="#2ca02c",
    )
    ax.plot(
        df.look_frac,
        df.blind,
        "-s",
        ms=4,
        color="#7f7f7f",
        lw=2,
        label="best blind schedule",
    )
    ax.fill_between(
        df.look_frac,
        df.blind - df.blind_ci,
        df.blind + df.blind_ci,
        alpha=0.15,
        color="#7f7f7f",
    )
    if be == be:
        ax.axvline(be, ls="--", color="k", lw=1)
        ax.text(be, ax.get_ylim()[1] * 0.95, f"  break-even {be:.2f}", fontsize=9)
    for x, c, lab in [
        (0.001, "#2ca02c", "load-avg\n(cheap)"),
        (0.93, "#d62728", "full telemetry\n(measured)"),
    ]:
        ax.axvline(x, color=c, lw=1.5, alpha=0.55)
        ax.text(
            x,
            ax.get_ylim()[1] * 0.45,
            f" {lab}",
            color=c,
            fontsize=8,
            ha="left" if x < 0.5 else "right",
        )
    ax.set_xlabel("cost of self-knowledge  (look-cost, fraction of a work-unit)")
    ax.set_ylabel("total life achievement")
    ax.set_title("The break-even price of self-knowledge")
    ax.legend(fontsize=8.5, loc="center left", frameon=False)
    fig.tight_layout()
    fig.savefig("fig1_breakeven.png", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote fig1_breakeven.png")


def make_sensitivity():
    df = pd.read_csv("study_sensitivity.csv")
    piv = df.pivot(index="wear", columns="cap", values="breakeven").sort_index(
        ascending=False
    )
    fig, ax = plt.subplots(figsize=(6, 4.4))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xlabel("RAM cap (MB)  [OOM cliff]")
    ax.set_ylabel("wear budget  [mortality cliff]")
    ax.set_title("Break-even price of self-knowledge: robustness")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="white" if v < 0.18 else "black",
                fontsize=9,
            )
    fig.colorbar(im, label="break-even look-cost")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig("fig2_sensitivity.png", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote fig2_sensitivity.png")


def main():
    make_hysteresis()
    make_breakeven()
    make_sensitivity()
    return 0


if __name__ == "__main__":
    sys.exit(main())
