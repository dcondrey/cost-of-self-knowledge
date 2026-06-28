"""Hardware confirmation of the contained-observer penalty.

Measure the CONTAINED projective frontier D(K)=K^2/2 on a real QPU and overlay the optimal POVM bound
D=(1-sqrt(1-K^2))/2 (external, dilation-allowed). The contained observer is a single qubit prepared in
|+>; it reads Z-information by a PROJECTIVE measurement in a basis tilted by beta from Z (mid-circuit),
K=cos(beta); the disturbance to |+> is read out by echo. No ancilla is used -- that is the point.

    /tmp/qenv/bin/python quantum_frontier.py sim    # Aer validation
    /tmp/qenv/bin/python quantum_frontier.py ibm    # real QPU
"""

import json
import logging
import math
import os
import sys

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("qfrontier")
SHOTS = 8192
BETAS = [0.20, 0.45, 0.70, 1.00, 1.30]  # tilt angles -> K = cos(beta) in (0,1)


def contained_circuit(beta):
    """Single qubit |+>; projective measurement in a basis tilted by beta from Z; echo. No ancilla."""
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(2, "c")  # c0 = projective outcome, c1 = echo
    qc = QuantumCircuit(q, c)
    qc.h(0)  # |+>
    qc.ry(-beta, 0)  # rotate the tilted measurement basis onto Z
    qc.measure(0, c[0])  # PROJECTIVE measurement (collapses; no ancilla)
    qc.ry(beta, 0)  # rotate back to the original frame
    qc.h(0)  # echo: |+> -> |0>
    qc.measure(0, c[1])
    return qc


def disturbance(counts):
    """D = 1 - P(echo == 0). c = 'c1 c0' (c1 = echo = leftmost char)."""
    tot = sum(counts.values())
    p0 = sum(v for k, v in counts.items() if k[0] == "0") / tot
    return 1 - p0


def run_aer(circuits):
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    sim = AerSimulator()
    return [
        sim.run(transpile(c, sim), shots=SHOTS).result().get_counts() for c in circuits
    ]


def run_ibm(circuits):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    tok = open(os.path.expanduser("~/.ibm_quantum_token")).read().strip()
    inst = open(os.path.expanduser("~/.ibm_quantum_instance")).read().strip()
    s = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok, instance=inst)
    backend = min(
        s.backends(simulator=False, operational=True),
        key=lambda b: b.status().pending_jobs,
    )
    log.info(f"backend = {backend.name}  pending = {backend.status().pending_jobs}")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa = [pm.run(c) for c in circuits]
    job = SamplerV2(mode=backend).run(isa, shots=SHOTS)
    log.info(f"job submitted: {job.job_id()} (waiting...)")
    res = job.result()
    return [r.data.c.get_counts() for r in res], backend.name


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"
    circuits = [contained_circuit(b) for b in BETAS]
    backend = "aer_simulator"
    if mode == "ibm":
        counts, backend = run_ibm(circuits)
    else:
        counts = run_aer(circuits)

    log.info(f"backend = {backend}  shots = {SHOTS}")
    log.info(
        f"{'beta':>6} {'K=cos b':>8} {'D_measured':>11} {'D=K^2/2':>9} {'D_optimal':>10}"
    )
    rows = []
    for beta, cnt in zip(BETAS, counts):
        K = math.cos(beta)
        D = disturbance(cnt)
        D_theory = K * K / 2
        D_opt = (1 - math.sqrt(1 - K * K)) / 2
        rows.append(
            {
                "beta": round(beta, 3),
                "K": round(K, 4),
                "D_measured": round(D, 4),
                "D_projective_theory": round(D_theory, 4),
                "D_optimal_povm": round(D_opt, 4),
                "penalty_vs_optimal": round(D - D_opt, 4),
            }
        )
        log.info(f"{beta:6.2f} {K:8.3f} {D:11.3f} {D_theory:9.3f} {D_opt:10.3f}")
    log.info(
        "=> measured contained frontier should track K^2/2, strictly above the optimal POVM bound; "
        "the gap is the self-measurement penalty on real hardware."
    )
    out = f"results/quantum-frontier-{'ibm' if mode == 'ibm' else 'sim'}.json"
    os.makedirs("results", exist_ok=True)
    json.dump(
        {"backend": backend, "shots": SHOTS, "rows": rows}, open(out, "w"), indent=2
    )
    log.info(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
