"""Theory: the contained-observer info-disturbance penalty, proven and computed.

A general measurement is a POVM; by Naimark's theorem a POVM requires an ancilla. A contained observer
(no fresh ancilla) is restricted to PROJECTIVE measurements of its own qubits. We compute, for extracting
Z-information K from a |+> qubit, the disturbance of (a) the optimal POVM / weak measurement (external,
dilation-allowed) and (b) the best projective measurement (contained, no dilation), and show the gap is
strictly positive for all partial information.

Result (verified numerically below):  D_external(K) = (1 - sqrt(1-K^2))/2     [optimal POVM / weak]
                                       D_contained(K) = K^2 / 2                [best projective]
The penalty Delta(K) = D_contained - D_external > 0 for 0<K<1, zero at K=0 and K=1 (where projective
Z is optimal). This is the operational, dilation-free form of Breuer's contained-observer limit.
"""

import logging
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("qtheory")

PLUS = np.array([1.0, 1.0]) / np.sqrt(2)
RHO = np.outer(PLUS, PLUS)
K0 = np.array([1.0, 0.0])
K1 = np.array([0.0, 1.0])


def fid(r):
    return float((PLUS @ r @ PLUS).real)


def weak(eps):
    """Optimal weak Z-measurement (square-root Kraus of POVM E_pm=(I ± eps Z)/2). Needs an ancilla."""
    a, b = np.sqrt((1 + eps) / 2), np.sqrt((1 - eps) / 2)
    Mp, Mm = np.diag([a, b]), np.diag([b, a])
    K = abs((1 + eps) / 2 - (1 - eps) / 2)  # Z-distinguishing power
    r2 = Mp @ RHO @ Mp.conj().T + Mm @ RHO @ Mm.conj().T
    return K, 1 - fid(r2)


def proj(beta):
    """Best projective measurement: read in a basis tilted by beta from Z toward X. No ancilla."""
    mp = np.array([np.cos(beta / 2), np.sin(beta / 2)])
    mm = np.array([-np.sin(beta / 2), np.cos(beta / 2)])
    Pp, Pm = np.outer(mp, mp), np.outer(mm, mm)
    K = abs(K0 @ Pp @ K0 - K1 @ Pp @ K1)
    r2 = Pp @ RHO @ Pp + Pm @ RHO @ Pm
    return K, 1 - fid(r2)


def main():
    ws = [weak(e) for e in np.linspace(0, 1, 60)]
    ps = [proj(b) for b in np.linspace(0, np.pi / 2, 60)]
    Kw, Dw = np.array(ws).T
    Kp, Dp = np.array(ps).T

    # verify the closed forms
    err_w = max(abs(D - (1 - np.sqrt(1 - K**2)) / 2) for K, D in ws)
    err_p = max(abs(D - K**2 / 2) for K, D in ps)
    log.info(f"closed-form check: external |D-(1-sqrt(1-K^2))/2| max = {err_w:.2e}")
    log.info(f"closed-form check: contained |D-K^2/2| max          = {err_p:.2e}")

    Kgrid = np.linspace(0, 1, 200)
    Dext = (1 - np.sqrt(1 - Kgrid**2)) / 2
    Dint = Kgrid**2 / 2
    delta = Dint - Dext
    kmax = Kgrid[int(np.argmax(delta))]
    log.info(
        f"penalty Delta(K) > 0 for all 0<K<1: min over interior = {delta[1:-1].min():.4f}"
    )
    log.info(f"max penalty Delta = {delta.max():.4f} at K = {kmax:.3f}")
    log.info(
        f"at K=0.5: external D={(1 - np.sqrt(1 - 0.25)) / 2:.4f}  contained D={0.25 / 2:.4f}  "
        f"Delta={0.25 / 2 - (1 - np.sqrt(0.75)) / 2:.4f}"
    )

    fig, ax = plt.subplots(figsize=(7, 3.9))
    ax.plot(
        Kgrid,
        Dext,
        label="external / optimal POVM (weak): $(1-\\sqrt{1-K^2})/2$",
        color="C0",
    )
    ax.plot(Kgrid, Dint, label="contained / projective only: $K^2/2$", color="C3")
    ax.fill_between(
        Kgrid,
        Dext,
        Dint,
        alpha=0.2,
        color="C3",
        label="self-measurement penalty $\\Delta(K)$",
    )
    import json as _json
    import os as _os

    if _os.path.exists("results/quantum-frontier-ibm.json"):
        hw = _json.load(open("results/quantum-frontier-ibm.json"))
        hk = [r["K"] for r in hw["rows"]]
        hd = [r["D_measured"] for r in hw["rows"]]
        ax.plot(
            hk,
            hd,
            "D",
            ms=7,
            color="black",
            label=f"real QPU ({hw['backend']})",
            zorder=5,
        )
    ax.plot(Kw, Dw, "o", ms=3, color="C0", alpha=0.5)
    ax.plot(Kp, Dp, "s", ms=3, color="C3", alpha=0.5)
    ax.set_xlabel("information extracted $K$ (Z-distinguishing power)")
    ax.set_ylabel("disturbance to the self $D$")
    ax.set_title("Contained-observer penalty: projective (no ancilla) vs optimal POVM")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig("fig4_quantum_penalty.png", dpi=150)
    plt.close(fig)
    log.info("wrote fig4_quantum_penalty.png")
    log.info(
        "=> Naimark: POVMs need an ancilla; a contained observer is projective-only; projective "
        "info-disturbance is strictly worse than optimal for partial info. The penalty is real, "
        "closed-form, and not reducible to the standard (dilation-assuming) bound."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
