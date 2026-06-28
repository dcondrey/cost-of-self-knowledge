"""Informational self-snapshot limit: the dynamic, measurable back-action complement to Breuer (1995).

Breuer ("The impossibility of accurate state self-measurements", 1995) proves a contained observer
cannot DISTINGUISH all states of the system it belongs to -- a static resolution limit. This measures
the DYNAMIC version: the act of self-measurement MUTATES the state it records, so a system cannot hold
a current, correct checksum of its own COMPLETE state, and the un-capturable core is bounded below by
the checksum apparatus's own footprint (a structural floor, not a thermodynamic one).

Construction: the complete self = STATE (a large body of bytes) + V (a fixed region holding a self-
checksum). We try to make V a correct checksum of the complete self, H(STATE + V):
  - EXTERNAL control: a checksum of STATE alone is consistent and stable (non-self IS capturable).
  - SELF: after V := H(STATE + V), H(STATE + V) != V -- storing the result invalidated it.
  - NON-CONVERGENCE: iterating V := H(STATE + V) never reaches a fixed point (V = H(STATE+V) is a
    hash-preimage condition, cryptographically unreachable) -- no calibrating it away (cf. the thermal
    hysteresis: the perturbation cannot be subtracted because subtracting it perturbs again).
  - FLOOR: everything EXCEPT the checksum region is captured faithfully; the irreducible un-capturable
    core is exactly |V|. Enlarging V to also cover itself only moves the core (regress), never to zero.
"""

import hashlib
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("snapshot")


def H(b, n=32):
    return hashlib.sha256(bytes(b)).digest()[:n]


def hamming(a, b):
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def run(state_bytes=1 << 20, vlen=32, iters=512):
    STATE = bytearray(os.urandom(state_bytes))
    V = bytearray(vlen)

    # EXTERNAL control: checksum of a region that EXCLUDES the checksum -> consistent + stable.
    E = H(STATE, vlen)
    external_consistent = H(STATE, vlen) == E

    # SELF: try to make V a correct checksum of the complete self (STATE + V).
    self_ever_consistent = False
    seen = set()
    fixed_point = False
    for _ in range(iters):
        c = H(STATE + V, vlen)  # checksum of the COMPLETE self, including current V
        if bytes(c) == bytes(V):  # V already equals checksum(STATE+V)?
            self_ever_consistent = True
        if bytes(V) in seen:  # any repeat => a cycle, i.e. it would have "settled"
            fixed_point = True
            break
        seen.add(bytes(V))
        V[:] = c  # store the result -> mutates V, part of the self

    # after storing, is the stored checksum current?
    current = H(STATE + V, vlen)
    final_consistent = bytes(current) == bytes(V)
    self_inconsistency_bits = hamming(
        current, V
    )  # how wrong the stored self-checksum is

    # FLOOR: STATE is captured faithfully; the un-capturable core is exactly the V region.
    captured_fraction = state_bytes / (state_bytes + vlen)

    log.info(f"complete self = STATE {state_bytes} B + V {vlen} B   (iters={iters})")
    log.info("-" * 64)
    log.info(
        f"EXTERNAL checksum (excludes itself) consistent : {external_consistent}   "
        f"(non-self IS capturable)"
    )
    log.info(
        f"SELF checksum ever consistent in {iters} iters   : {self_ever_consistent}"
    )
    log.info(f"SELF checksum reached a fixed point             : {fixed_point}")
    log.info(
        f"final stored self-checksum is current           : {final_consistent}   "
        f"(off by {self_inconsistency_bits}/{vlen * 8} bits ~ random)"
    )
    log.info(
        f"fraction of self captured faithfully            : {captured_fraction * 100:.4f}%  "
        f"(everything but the {vlen} B apparatus)"
    )
    log.info("-" * 64)
    if external_consistent and not self_ever_consistent:
        log.info(
            "=> A system CAN faithfully checksum any part of itself EXCEPT the checksum apparatus. "
            "The un-capturable core is the apparatus's own footprint, bounded below by |V| and "
            "never reachable -- the act of self-measurement invalidates its own result. Dynamic, "
            "measurable back-action; the informational complement to Breuer (1995)."
        )
    else:
        log.info(
            "=> unexpected: self-snapshot converged or was consistent (investigate)."
        )
    return {
        "external_consistent": external_consistent,
        "self_ever_consistent": self_ever_consistent,
        "fixed_point": fixed_point,
        "final_consistent": final_consistent,
        "inconsistency_bits": self_inconsistency_bits,
        "vlen": vlen,
        "state_bytes": state_bytes,
    }


def floor_sweep():
    log.info(
        "\nFLOOR: the un-capturable core equals the apparatus size |V| (and is never 0):"
    )
    for vlen in (16, 32, 64):
        logging.disable(logging.INFO)
        r = run(vlen=vlen, iters=128)
        logging.disable(logging.NOTSET)
        core = r["vlen"] if not r["self_ever_consistent"] else 0
        log.info(f"  |V|={vlen:3d} B -> un-capturable core = {core} B (never 0)")


def main():
    run()
    floor_sweep()
    return 0


if __name__ == "__main__":
    sys.exit(main())
