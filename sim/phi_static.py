"""phi_static validation gate (ASDS, EXPERIMENT.md Section A.5/A.6).

Confirms the committed Section A constants produce the registered targets BEFORE any
RL is built:
  - violation base-rate in 10-15%,
  - phi_static^H ~= 0.5  (irreducibly hidden fraction),
  - phi_static^L ~= 0    (latently reconstructable -> fully absorbable).

phi_static = H(u_{t+1} | O_t-history) / H(u_{t+1}) under the stationary distribution.
Two independent estimates, which must agree:
  (a) analytic steady-state Kalman on the linear-Gaussian approximation (u as a linear
      measurement of z with quantization noise);
  (b) empirical: a linear least-squares predictor of z_{t+1} from a finite observable
      history window (previews the absorber, needs no training).

Exits non-zero if any committed target is missed, so it functions as a real gate.
"""
import logging
import math
import sys
import numpy as np

from env import EnvConfig, rollout, reference_actions, throttle

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("phi_static")

SEED = 20240623
N_STEPS = 200_000
HIST = 32                      # observable history window for the empirical estimate
TARGET_PHI_H = 0.50
PHI_H_TOL = 0.07
C_Q = 1.175                   # z_crit = mean_z + C_Q*std_z  -> ~12% violation base-rate
BASE_RATE_LO, BASE_RATE_HI = 0.10, 0.15


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def quantized_probs(mean, std, cfg: EnvConfig):
    """P(u=k) for k=0..K-1 when z ~ N(mean, std^2), matching env.throttle binning."""
    delta = cfg.z_crit / (cfg.K - 1)
    edges = [(k + 0.5) * delta for k in range(cfg.K - 1)]
    std = max(std, 1e-9)
    cdf = [_phi((e - mean) / std) for e in edges]
    probs = np.empty(cfg.K)
    probs[0] = cdf[0]
    for k in range(1, cfg.K - 1):
        probs[k] = cdf[k] - cdf[k - 1]
    probs[cfg.K - 1] = 1.0 - cdf[-1]
    return np.clip(probs, 0.0, 1.0)


def _entropy(probs):
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def grid_filter_phi(traj, cfg: EnvConfig, n_sub=3000, grid_n=200):
    """Optimal recursive Bayesian filter of z on a grid, with the exact quantized-u likelihood.

    u_t deterministically reveals z_t's bin (measurement update = truncate belief to that
    bin). Between steps belief spreads by sigma_w only; a_t, x_t are known inputs folded into
    the drift. Returns phi = mean_t H(u_{t+1}|O_t) / H(u). Variant L (sigma_w=0) keeps the
    belief a point mass -> phi -> 0. No linear-Gaussian approximation of the quantizer.
    """
    x, a, u, z = traj["x"], traj["a"], traj["u"], traj["z"]
    mean_z, std_z = float(np.mean(z)), float(np.std(z))
    delta = cfg.z_crit / (cfg.K - 1)            # grid must span all K bins, not just mean +/- 6 std
    lo = min(mean_z - 6 * std_z, -0.5 * delta)
    hi = max(mean_z + 6 * std_z, (cfg.K - 0.5) * delta)
    grid = np.linspace(lo, hi, grid_n)
    bin_of = np.clip(np.round((cfg.K - 1) * grid / cfg.z_crit), 0, cfg.K - 1).astype(int)
    masks = [bin_of == k for k in range(cfg.K)]
    sw = max(cfg.sigma_w, 1e-3 * std_z)
    norm = 1.0 / (sw * math.sqrt(2 * math.pi))
    drift = cfg.lam * a + cfg.mu * x

    h_marg = _entropy(quantized_probs(mean_z, std_z, cfg))
    prior = np.exp(-0.5 * ((grid - mean_z) / std_z) ** 2)
    prior /= prior.sum()
    belief = prior.copy()
    h_acc, count = 0.0, 0
    for t in range(min(n_sub, len(z)) - 1):
        if t > 0:                               # belief = p(z_t|O_{t-1}); its bin-spread = H(u_t|O_{t-1})
            binprobs = np.array([belief[m].sum() for m in masks])
            h_acc += _entropy(np.clip(binprobs, 0.0, 1.0))
            count += 1
        bin_mask = masks[int(u[t])]
        post = belief * bin_mask                # measurement update: u_t reveals the bin
        s = post.sum()
        if s > 0:
            post = post / s
        elif bin_mask.sum() > 0:
            post = bin_mask / bin_mask.sum()
        else:
            post = belief                       # observed bin off-grid: measurement uninformative
        means = cfg.beta * grid + drift[t]      # predict z_{t+1}: drift known, spread by sigma_w
        kernel = norm * np.exp(-0.5 * ((grid[None, :] - means[:, None]) / sw) ** 2)
        belief = post @ kernel
        bsum = belief.sum()
        belief = belief / bsum if bsum > 0 and np.isfinite(bsum) else prior.copy()
    return (h_acc / count) / h_marg if count and h_marg > 0 else 0.0


def empirical_phi(traj, cfg: EnvConfig):
    """phi lower-bound from a linear lstsq predictor of z_{t+1} on observable history.

    Features at t: [x_{t-h..t}, a_{t-h..t}, u_{t-h..t}, 1]. Residual variance of z_{t+1}
    after regression -> predictive entropy of u_{t+1} -> phi. A linear preview of the
    absorber; the true closed-loop absorber will tighten this.
    """
    x, a, u, z = traj["x"], traj["a"], traj["u"], traj["z"]
    n = len(z)
    rows, target = [], []
    for t in range(HIST, n - 1):
        feat = np.concatenate([x[t - HIST:t + 1], a[t - HIST:t + 1],
                               u[t - HIST:t + 1].astype(float), [1.0]])
        rows.append(feat)
        target.append(z[t + 1])
    A = np.asarray(rows)
    y = np.asarray(target)
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    pred_var = float(np.var(resid))
    var_z, mean_z = float(np.var(z)), float(np.mean(z))
    h_marg = _entropy(quantized_probs(mean_z, math.sqrt(var_z), cfg))
    means = (A @ coef)
    sel = np.random.default_rng(0).choice(len(means), size=min(4000, len(means)), replace=False)
    h_cond = float(np.mean([_entropy(quantized_probs(float(m), math.sqrt(pred_var), cfg))
                            for m in means[sel]]))
    return (h_cond / h_marg if h_marg > 0 else 0.0), pred_var


def calibrate(base_cfg: EnvConfig, sigma_w: float, rng: np.random.Generator):
    """For a candidate sigma_w: simulate variant H, set z_crit = 1.5*std(z), report stats."""
    cfg = EnvConfig(**{**base_cfg.__dict__, "sigma_w": sigma_w})
    actions = reference_actions(N_STEPS, rng)
    traj = rollout(cfg, "H", actions, rng)
    std_z = float(np.std(traj["z"]))
    mean_z = float(np.mean(traj["z"]))
    cfg = EnvConfig(**{**cfg.__dict__, "z_crit": mean_z + C_Q * std_z})
    traj = rollout(cfg, "H", actions, rng)        # recompute u, V with calibrated z_crit
    base_rate = float(np.mean(traj["V"]))
    phi_k = grid_filter_phi(traj, cfg)
    return cfg, traj, base_rate, phi_k


def main():
    rng = np.random.default_rng(SEED)
    base = EnvConfig()

    lo, hi = 0.02, 1.5                # bisection on sigma_w to hit phi_static^H = target
    cfg = traj = None
    base_rate = phi_k = float("nan")
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        cfg, traj, base_rate, phi_k = calibrate(base, mid, rng)
        if phi_k < TARGET_PHI_H:
            lo = mid                  # more hidden noise -> higher phi
        else:
            hi = mid
        if abs(phi_k - TARGET_PHI_H) < 1e-3:
            break

    phi_emp, _ = empirical_phi(traj, cfg)

    cfg_L = cfg.for_variant("L")
    traj_L = rollout(cfg_L, "L", traj["a"], rng)
    phi_k_L = grid_filter_phi(traj_L, cfg_L)
    phi_emp_L, _ = empirical_phi(traj_L, cfg_L)

    log.info("=" * 64)
    log.info("ASDS phi_static validation gate  (EXPERIMENT.md Section A)")
    log.info("=" * 64)
    log.info(f"  tuned sigma_w        = {cfg.sigma_w:.4f}")
    log.info(f"  calibrated z_crit    = {cfg.z_crit:.4f}  (= mean + {C_Q} * std(z))")
    log.info(f"  violation base-rate  = {base_rate*100:.2f}%   target 10-15%")
    log.info(f"  phi_static^H filter  = {phi_k:.3f}     target ~{TARGET_PHI_H:.2f}")
    log.info(f"  phi_static^H lstsq   = {phi_emp:.3f}     (cross-check, should be >= filter)")
    log.info(f"  phi_static^L filter  = {phi_k_L:.3f}     target ~0")
    log.info(f"  phi_static^L lstsq   = {phi_emp_L:.3f}     target ~0")
    log.info("-" * 64)

    checks = {
        "base-rate in 10-15%": BASE_RATE_LO <= base_rate <= BASE_RATE_HI,
        "phi_H ~= target": abs(phi_k - TARGET_PHI_H) <= PHI_H_TOL,
        "phi_H estimates agree": abs(phi_k - phi_emp) <= 0.15,
        "phi_L ~= 0 (Kalman)": phi_k_L <= 0.05,
        "phi_L ~= 0 (lstsq)": phi_emp_L <= 0.15,
    }
    for name, ok in checks.items():
        log.info(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    log.info("=" * 64)

    if not all(checks.values()):
        log.info("GATE FAILED: tune Section A constants on paper before building the RL loop.")
        return 1
    log.info("GATE PASSED: Section A constants validated. Commit z_crit and sigma_w above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
