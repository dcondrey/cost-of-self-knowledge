"""Fatigue-limited throughput environment (ASDS, EXPERIMENT.md Section A).

Shared, registered environment for BOTH the phi_static validation (phi_static.py)
and the later closed-loop active-A experiment. Do not fork a convenience copy.

Variants differ in exactly one parameter, the hidden-fatigue innovation noise:
  - 'H' (irreducibly hidden):      sigma_w > 0, exogenous, no causal path to A's channel.
  - 'L' (latently reconstructable): sigma_w = 0, so z_t is a deterministic function of
                                    (z_0, a_<t, x_<t), all of which A possesses.

Dynamics (Section A.1):
  x_{t+1} = alpha x_t + kappa a_t + eta_t      eta ~ N(0, sigma_x^2)   (observable stress)
  z_{t+1} = beta  z_t + lam   a_t + mu x_t + w_t  w ~ N(0, sigma_w^2)  (hidden fatigue)
  V_t = 1[z_t > z_crit]                                                (violation/damage)
  u_t = clip(round((K-1) z_t / z_crit), 0, K-1)                        (B's faithful report)
  r_t = a_t (1 - c u_t/(K-1))                                          (throttled throughput)
"""
from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    alpha: float = 0.8
    kappa: float = 1.0
    sigma_x: float = 0.1
    beta: float = 0.9
    lam: float = 0.5
    mu: float = 0.3
    sigma_w: float = 0.8352  # variant H innovation; validated by phi_static.py -> phi_static^H=0.50. L forces 0.
    K: int = 5
    c: float = 0.5
    z_crit: float = 12.59   # validated: mean_z + 1.175*std_z under reference policy -> 11.9% base-rate
    damage_penalty: float = 2.0  # boundary-criticality: cost of a violation, forces closed-loop z-tracking
    horizon: int = 200

    def for_variant(self, variant: str) -> "EnvConfig":
        if variant == "L":
            return replace(self, sigma_w=0.0)
        if variant == "H":
            return self
        raise ValueError(f"variant must be 'H' or 'L', got {variant!r}")


def throttle(z, cfg: EnvConfig):
    """B's faithful quantized report of fatigue z. Scalar or array."""
    q = np.round((cfg.K - 1) * np.asarray(z) / cfg.z_crit)
    return np.clip(q, 0, cfg.K - 1).astype(int)


def rollout(cfg: EnvConfig, variant: str, actions, rng: np.random.Generator):
    """Run the environment for len(actions) steps under a fixed action sequence.

    Returns dict of per-step arrays (length N): x, z, a, u, V, r.
    x[t], z[t] are the states observed at step t (u[t], V[t] computed from them);
    the action a[t] then drives the transition to step t+1.
    """
    vcfg = cfg.for_variant(variant)
    a = np.clip(np.asarray(actions, dtype=float), 0.0, 1.0)
    n = a.shape[0]
    x = np.empty(n)
    z = np.empty(n)
    eta = rng.normal(0.0, vcfg.sigma_x, size=n)
    w = rng.normal(0.0, vcfg.sigma_w, size=n) if vcfg.sigma_w > 0 else np.zeros(n)
    xt, zt = 0.0, 0.0
    for t in range(n):
        x[t] = xt
        z[t] = zt
        xt = vcfg.alpha * xt + vcfg.kappa * a[t] + eta[t]
        zt = vcfg.beta * zt + vcfg.lam * a[t] + vcfg.mu * x[t] + w[t]
    u = throttle(z, vcfg)
    V = (z > vcfg.z_crit).astype(int)
    r = a * (1.0 - vcfg.c * u / (vcfg.K - 1)) - vcfg.damage_penalty * V
    return {"x": x, "z": z, "a": a, "u": u, "V": V, "r": r}


def reference_actions(n, rng: np.random.Generator):
    """Reference exploration policy for stationary statistics: a ~ Uniform[0,1].

    Untrained/exploring A. Used only to define the stationary distribution against
    which phi_static and the violation base-rate are calibrated.
    """
    return rng.uniform(0.0, 1.0, size=n)


def batch_step(vcfg: EnvConfig, x, z, a, rng):
    """Vectorized one-step transition over a batch of trajectories (variant-resolved vcfg).

    Same dynamics as rollout()/Stepper, applied to arrays. Returns
    (u, V, reward, x_next, z_next). Used by the RL training/eval inner loop for speed.
    """
    a = np.clip(a, 0.0, 1.0)
    u = np.clip(np.round((vcfg.K - 1) * z / vcfg.z_crit), 0, vcfg.K - 1).astype(int)
    V = (z > vcfg.z_crit).astype(int)
    reward = a * (1.0 - vcfg.c * u / (vcfg.K - 1)) - vcfg.damage_penalty * V
    eta = rng.normal(0.0, vcfg.sigma_x, size=x.shape)
    w = rng.normal(0.0, vcfg.sigma_w, size=x.shape) if vcfg.sigma_w > 0 else 0.0
    x_next = vcfg.alpha * x + vcfg.kappa * a + eta
    z_next = vcfg.beta * z + vcfg.lam * a + vcfg.mu * x + w
    return u, V, reward, x_next, z_next


class Stepper:
    """Stateful single-trajectory driver for policy rollouts (same dynamics as rollout()).

    step(a) applies B's throttle to the current fatigue, returns the throttled reward, then
    advances the dynamics. The returned obs exposes x (observable stress) and u (the throttle
    just applied); z is returned too but is A-hidden (callers must not feed it to the policy
    except in the oracle control).
    """

    def __init__(self, cfg: EnvConfig, variant: str, rng: np.random.Generator):
        self.cfg = cfg.for_variant(variant)
        self.rng = rng
        self.reset()

    def reset(self):
        self.x, self.z, self.t = 0.0, 0.0, 0
        return {"x": self.x, "u": 0, "V": 0, "z": self.z}

    def step(self, a):
        c = self.cfg
        a = float(np.clip(a, 0.0, 1.0))
        u = int(np.clip(round((c.K - 1) * self.z / c.z_crit), 0, c.K - 1))
        V = int(self.z > c.z_crit)
        reward = a * (1.0 - c.c * u / (c.K - 1)) - c.damage_penalty * V
        eta = self.rng.normal(0.0, c.sigma_x)
        w = self.rng.normal(0.0, c.sigma_w) if c.sigma_w > 0 else 0.0
        x_prev = self.x
        self.x = c.alpha * self.x + c.kappa * a + eta
        self.z = c.beta * self.z + c.lam * a + c.mu * x_prev + w
        self.t += 1
        done = self.t >= c.horizon
        return {"x": self.x, "u": u, "V": V, "z": self.z, "reward": reward, "done": done}
