"""Active-agent smoke test (ASDS milestone 3a, batched API).

Fast single-seed check that the batched REINFORCE agent learns and the grip/collapse contrast
points the right way, before the multi-seed sweep in experiment.py.
"""
import logging
import sys
import numpy as np

from env import EnvConfig, rollout, reference_actions
from agent import FeatCfg, train_agent, eval_agent
from absorber import measure_phi_effective

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("validate_agent")

SEED = 11
M, HIDDEN = 8, 64


def random_return(cfg, variant, rng):
    tr = rollout(cfg.for_variant(variant), variant, reference_actions(40_000, rng), rng)
    return float(np.mean(tr["r"])) * cfg.horizon


def run_variant(cfg, variant, rng):
    rnd = random_return(cfg, variant, rng)
    fc_h, fc_o = FeatCfg(M, oracle=False), FeatCfg(M, oracle=True)
    pol_h, mu_h, sd_h = train_agent(cfg, variant, fc_h, HIDDEN, seed=SEED)
    pol_o, mu_o, sd_o = train_agent(cfg, variant, fc_o, HIDDEN, seed=SEED)
    ret_h, vr_h, traj_h = eval_agent(pol_h, cfg, variant, fc_h, mu_h, sd_h, seed=SEED)
    ret_o, _, _ = eval_agent(pol_o, cfg, variant, fc_o, mu_o, sd_o, seed=SEED)
    phi, _, _ = measure_phi_effective(traj_h, M, [128, 128], cfg.K, seed=SEED)
    return {"random": rnd, "hidden": ret_h, "oracle": ret_o,
            "grip": ret_o - ret_h, "vrate": vr_h, "phi_eff": phi}


def main():
    rng = np.random.default_rng(SEED)
    cfg = EnvConfig()
    res = {v: run_variant(cfg, v, rng) for v in ("H", "L")}

    log.info("=" * 74)
    log.info("ASDS active-agent smoke test (milestone 3a, batched)")
    log.info("=" * 74)
    log.info(f"{'variant':8s} {'random':>8s} {'hidden':>8s} {'oracle':>8s} {'grip':>8s} "
             f"{'vrate':>7s} {'phi_eff':>9s}")
    for v in ("H", "L"):
        r = res[v]
        log.info(f"{v:8s} {r['random']:8.2f} {r['hidden']:8.2f} {r['oracle']:8.2f} "
                 f"{r['grip']:8.2f} {r['vrate']*100:6.1f}% {r['phi_eff']:9.3f}")
    log.info("-" * 74)

    H, L = res["H"], res["L"]
    checks = {
        "agent learns (H, L > random)": H["hidden"] > H["random"] and L["hidden"] > L["random"],
        "hidden-value grip in H (> 0)": H["grip"] > 0.3,
        "grip smaller in L than H": L["grip"] < H["grip"],
        "active A does NOT collapse phi^H (>0.40)": H["phi_eff"] > 0.40,
        "active A collapses phi^L (<0.25)": L["phi_eff"] < 0.25,
    }
    for name, ok in checks.items():
        log.info(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    log.info("=" * 74)
    ok = all(checks.values())
    log.info("MILESTONE 3a PASSED (batched)." if ok else "MILESTONE 3a FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
