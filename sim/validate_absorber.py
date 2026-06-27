"""Closed-loop absorber validation (ASDS milestone 2).

Confirms the trained absorber's phi_effective reproduces the analytic phi_static under a
RANDOM agent (reference policy), before any RL exists:
  - variant H: phi_effective^H ~= phi_static^H ~= 0.50  (hidden noise is irreducible),
  - variant L: phi_effective^L -> small and shrinks with absorber capacity (z is
    reconstructable from A's own action history; the collapse is capacity-driven, not luck).

This is the measurement plumbing the RL experiment depends on. If it does not reproduce
phi_static here, the closed-loop numbers later are not trustworthy.
"""
import logging
import sys
import numpy as np

from env import EnvConfig, rollout, reference_actions
from absorber import measure_phi_effective

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("validate_absorber")

SEED = 7
N_STEPS = 60_000
PHI_STATIC_H = 0.50
SIZES = [("m=8  small", 8, [64, 64]),
         ("m=32 small", 32, [64, 64]),
         ("m=32 large", 32, [256, 256])]


def main():
    rng = np.random.default_rng(SEED)
    cfg = EnvConfig()
    actions = reference_actions(N_STEPS, rng)
    traj_H = rollout(cfg, "H", actions, rng)
    traj_L = rollout(cfg.for_variant("L"), "L", actions, rng)

    log.info("=" * 68)
    log.info("ASDS closed-loop absorber validation (milestone 2, random agent)")
    log.info("=" * 68)
    log.info(f"{'absorber':12s} {'params':>8s}   phi_eff^H   phi_eff^L")
    log.info("-" * 68)
    phi_H, phi_L = {}, {}
    for name, m, hidden in SIZES:
        pH, _, npar = measure_phi_effective(traj_H, m, hidden, cfg.K, seed=SEED)
        pL, _, _ = measure_phi_effective(traj_L, m, hidden, cfg.K, seed=SEED)
        phi_H[name], phi_L[name] = pH, pL
        log.info(f"{name:12s} {npar:8d}   {pH:.3f}       {pL:.3f}")
    log.info("-" * 68)

    big = SIZES[-1][0]
    checks = {
        "phi_eff^H ~= phi_static (0.50)": abs(phi_H[big] - PHI_STATIC_H) <= 0.10,
        "phi_eff^L collapses (< 0.20)": phi_L[big] < 0.20,
        "L collapse is capacity-driven": phi_L[big] <= phi_L[SIZES[0][0]] + 0.02,
        "H >> L at large capacity": phi_H[big] - phi_L[big] > 0.25,
    }
    for name, ok in checks.items():
        log.info(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    log.info("=" * 68)
    if not all(checks.values()):
        log.info("MILESTONE 2 FAILED: absorber does not reproduce phi_static; fix before RL.")
        return 1
    log.info("MILESTONE 2 PASSED: phi_effective measurement validated against phi_static.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
