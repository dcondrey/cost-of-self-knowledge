"""Quantum self-measurement on real hardware (IBM Quantum) -- the quantum regime of the self-measurement
back-action, parallel to selfmeasure.py (thermal) and snapshot.py (informational).

Experiment 1 -- BACK-ACTION SWEEP: a self-monitor qubit couples to a body qubit with strength theta and
is then read; we measure the body's disturbance (loss of X-coherence) vs theta. The quantum analog of
ramping self-measurement intensity: disturbance rises with the information extracted.

Experiment 2 -- SELF-MEASUREMENT PENALTY: a 2-qubit "self" {q0,q1}. We read q0 with an EXTERNAL fresh
ancilla, vs with the INTERNAL qubit q1 (itself part of the self). For matched information about q0, the
internal readout disturbs MORE of the self -- a contained observer has no outside apparatus and must
collapse part of itself to obtain a classical self-report. Delta = D_internal - D_external is the cost
of being inside: the operational, hardware-measured form of Breuer (1995)'s contained-observer limit,
with a bounded-below floor (the sacrificed apparatus qubit). Physics (measurement back-action) is
textbook; the self-measurement framing and the penalty quantity are the contribution.

    /tmp/qenv/bin/python quantum_selfmeasure.py sim    # Aer validation
    /tmp/qenv/bin/python quantum_selfmeasure.py ibm    # real QPU (token in ~/.ibm_quantum_token)
"""

import json
import logging
import math
import os
import sys

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("qsm")

SHOTS = 8192
THETAS = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi]


def penalty_circuit(theta, mode):
    """self = {q0,q1} each |+>. Read q0 with monitor (ancilla q2 if external, q1 if internal); then
    echo (H,H) the self and read it. c[0]=q0 echo, c[1]=q1 echo, c[2]=monitor."""
    n = 3 if mode == "external" else 2
    q = QuantumRegister(n, "q")
    c = ClassicalRegister(3, "c")
    qc = QuantumCircuit(q, c)
    qc.h(0)
    qc.h(1)  # the self in |+>|+>
    mon = 2 if mode == "external" else 1
    if mode == "internal":
        qc.reset(
            mon
        )  # no fresh ancilla: must ERASE a self-qubit to get a usable pointer
    qc.cry(theta, 0, mon)  # monitor acquires which-path info about q0
    qc.measure(mon, c[2])  # the readout (collapse)
    qc.h(0)
    qc.h(1)  # echo U^dag on the self
    qc.measure(0, c[0])
    qc.measure(1, c[1])
    return qc


def analyse(counts):
    tot = sum(counts.values())
    # qiskit keys are big-endian over c: "c2 c1 c0" (no spaces, single creg) -> key[::-1] = c0 c1 c2
    self_intact = sum(
        v for k, v in counts.items() if k[-1] == "0" and k[-2] == "0"
    )  # c0==0 and c1==0
    mon_one = sum(v for k, v in counts.items() if k[-3] == "1")  # c2==1
    disturbance = 1.0 - self_intact / tot
    info = (
        mon_one / tot
    )  # P(monitor=1): which-path info about q0 (matched across modes)
    return disturbance, info


def run_aer(circuits):
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    sim = AerSimulator()
    out = []
    for qc in circuits:
        res = sim.run(transpile(qc, sim), shots=SHOTS).result().get_counts()
        out.append(res)
    return out


def run_ibm(circuits):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    token = open(os.path.expanduser("~/.ibm_quantum_token")).read().strip()
    inst_path = os.path.expanduser("~/.ibm_quantum_instance")
    instance = open(inst_path).read().strip() if os.path.exists(inst_path) else None
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform", token=token, instance=instance
    )
    backend = service.least_busy(operational=True, simulator=False)
    log.info(f"backend = {backend.name}")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa = [pm.run(qc) for qc in circuits]
    sampler = SamplerV2(mode=backend)
    job = sampler.run(isa, shots=SHOTS)
    log.info(
        f"job submitted: {job.job_id()} (queue + run; this can take a while on the free tier)"
    )
    res = job.result()
    return [r.data.c.get_counts() for r in res], backend.name


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"
    circuits, labels = [], []
    for th in THETAS:
        circuits.append(penalty_circuit(th, "external"))
        labels.append(("external", th))
        circuits.append(penalty_circuit(th, "internal"))
        labels.append(("internal", th))
    backend = "aer_simulator"
    if mode == "ibm":
        counts_list, backend = run_ibm(circuits)
    else:
        counts_list = run_aer(circuits)

    rows = []
    for (m, th), counts in zip(labels, counts_list):
        d, info = analyse(counts)
        rows.append(
            {
                "mode": m,
                "theta": round(th, 4),
                "disturbance": round(d, 4),
                "info": round(info, 4),
            }
        )
    log.info(f"backend = {backend}   shots = {SHOTS}")
    log.info(
        f"{'theta':>7} {'info(q0)':>9} {'D_external':>11} {'D_internal':>11} {'penalty':>8}"
    )
    for th in THETAS:
        e = next(
            r for r in rows if r["mode"] == "external" and abs(r["theta"] - th) < 1e-3
        )
        i = next(
            r for r in rows if r["mode"] == "internal" and abs(r["theta"] - th) < 1e-3
        )
        log.info(
            f"{th:7.3f} {i['info']:9.3f} {e['disturbance']:11.3f} {i['disturbance']:11.3f} "
            f"{i['disturbance'] - e['disturbance']:8.3f}"
        )
    e = next(
        r for r in rows if r["mode"] == "external" and abs(r["theta"] - math.pi) < 1e-3
    )
    i = next(
        r for r in rows if r["mode"] == "internal" and abs(r["theta"] - math.pi) < 1e-3
    )
    delta = round(i["disturbance"] - e["disturbance"], 4)
    log.info("-" * 50)
    log.info(
        f"PENALTY at matched FULL info (theta=pi): external D={e['disturbance']:.3f}  "
        f"internal D={i['disturbance']:.3f}  Delta={delta:.3f}"
    )
    log.info(
        "  external preserves its apparatus via a FRESH ancilla; the contained observer has none and "
        "must spend a self-qubit. Delta is the no-dilation self-measurement penalty -- not captured by "
        "standard (dilation-assuming) info-disturbance bounds. (theta=0 is degenerate: zero info.)"
    )
    result = {
        "backend": backend,
        "shots": SHOTS,
        "penalty_matched_full": delta,
        "rows": rows,
    }
    out = f"results/quantum-{'ibm' if mode == 'ibm' else 'sim'}.json"
    os.makedirs("results", exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)
    log.info(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
