"""Minimal hardware run: only the matched-full-info penalty (theta=pi, external vs internal) on the
least-busy QPU -- the fastest path to the headline Delta. The full theta sweep stays simulator-only
(it only shapes the curve); hardware just needs to confirm the headline point. Optionally cancels a
prior job id (argv[1]) to free our queue slot.

    /tmp/qenv/bin/python quantum_fast.py [old_job_id_to_cancel]
"""

import json
import logging
import math
import os
import sys

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from quantum_selfmeasure import analyse, penalty_circuit

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("qfast")
SHOTS = 8192


def main():
    tok = open(os.path.expanduser("~/.ibm_quantum_token")).read().strip()
    inst = open(os.path.expanduser("~/.ibm_quantum_instance")).read().strip()
    s = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok, instance=inst)

    if len(sys.argv) > 1:
        try:
            s.job(sys.argv[1]).cancel()
            log.info(f"cancelled prior job {sys.argv[1]}")
        except (Exception,) as e:
            log.info(f"cancel note: {str(e)[:100]}")

    backend = min(
        s.backends(simulator=False, operational=True),
        key=lambda b: b.status().pending_jobs,
    )
    log.info(f"backend = {backend.name}  pending = {backend.status().pending_jobs}")
    circs = [penalty_circuit(math.pi, "external"), penalty_circuit(math.pi, "internal")]
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa = [pm.run(c) for c in circs]
    job = SamplerV2(mode=backend).run(isa, shots=SHOTS)
    log.info(f"minimal job submitted: {job.job_id()} (waiting for result...)")
    res = job.result()
    de, ie = analyse(res[0].data.c.get_counts())
    di, ii = analyse(res[1].data.c.get_counts())
    delta = round(di - de, 4)
    log.info("=" * 56)
    log.info(f"HARDWARE penalty on {backend.name} ({SHOTS} shots):")
    log.info(f"  external D = {de:.3f}  (info {ie:.3f})")
    log.info(f"  internal D = {di:.3f}  (info {ii:.3f})")
    log.info(f"  Delta = {delta:.3f}   (sim was 0.25; noise shrinks it)")
    out = {
        "backend": backend.name,
        "shots": SHOTS,
        "external_D": round(de, 4),
        "internal_D": round(di, 4),
        "external_info": round(ie, 4),
        "internal_info": round(ii, 4),
        "delta": delta,
    }
    json.dump(out, open("results/quantum-ibm.json", "w"), indent=2)
    log.info("wrote results/quantum-ibm.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
