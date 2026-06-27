"""Active agent A (ASDS, EXPERIMENT.md Section 3, milestone 3).

REINFORCE agent that optimizes throttled throughput under a damage penalty (boundary-critical,
so closed-loop z-tracking matters). Batched (vectorized over parallel trajectories) for speed.

Three feature switches drive the experiment:
  - oracle:  policy additionally sees the hidden fatigue z (fully-informed reference for grip).
  - sees_u:  policy observes B's throttle u (its bin leaks z). Toggling this off tests whether
             B's own faithful intervention is what de-hides z (the engagement-dilemma leak).
"""
from dataclasses import dataclass
import numpy as np

from env import EnvConfig, rollout, reference_actions, batch_step


@dataclass(frozen=True)
class FeatCfg:
    m: int = 8
    oracle: bool = False
    sees_u: bool = True

    def in_dim(self):
        return self.m + self.m + (self.m if self.sees_u else 0) + (1 if self.oracle else 0)


def _feature_block(Xh, Ah, Uh, zc, fc: FeatCfg):
    """Assemble the (batched) raw feature matrix [B, in_dim] from history arrays [B, m]."""
    parts = [Xh, Ah]
    if fc.sees_u:
        parts.append(Uh)
    if fc.oracle:
        parts.append(zc[:, None])
    return np.concatenate(parts, axis=1)


class Policy:
    """1-hidden-layer MLP, scalar pre-activation -> sigmoid mean in (0,1). Adam."""

    def __init__(self, in_dim, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((in_dim, hidden)) * np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, 1)) * 0.01).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self._p = [self.W1, self.b1, self.W2, self.b2]
        self.m = [np.zeros_like(p) for p in self._p]
        self.v = [np.zeros_like(p) for p in self._p]
        self.t = 0

    def n_params(self):
        return sum(p.size for p in self._p)

    def mean(self, F):
        H = np.maximum(0.0, F @ self.W1 + self.b1)
        mu = 1.0 / (1.0 + np.exp(-(H @ self.W2 + self.b2)))
        return mu[:, 0]

    def policy_grad_step(self, F, a, mu, adv, sigma, lr=3e-3):
        H = np.maximum(0.0, F @ self.W1 + self.b1)
        do = (adv * (a - mu) / sigma ** 2 * mu * (1.0 - mu))[:, None]
        gW2 = -H.T @ do
        gb2 = -do.sum(0)
        dH = (-do @ self.W2.T) * (H > 0)
        gW1 = F.T @ dH
        gb1 = dH.sum(0)
        grads = [gW1, gb1, gW2, gb2]
        gnorm = np.sqrt(sum(float((g * g).sum()) for g in grads))   # global-norm clip: REINFORCE stability
        if gnorm > 5.0:
            grads = [g * (5.0 / gnorm) for g in grads]
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(self._p, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mhat = self.m[i] / (1 - b1 ** self.t)
            vhat = self.v[i] / (1 - b2 ** self.t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)


def _norm_stats(cfg, variant, fc: FeatCfg, rng):
    """Feature normalization from a reference random rollout (policy-independent, stable)."""
    tr = rollout(cfg.for_variant(variant), variant, reference_actions(20_000, rng), rng)
    x, a, u, z = tr["x"], tr["a"], tr["u"], tr["z"]
    rows = []
    for t in range(fc.m, len(x)):
        rows.append(_feature_block(x[t - fc.m:t][None], a[t - fc.m:t][None],
                                   u[t - fc.m:t].astype(float)[None],
                                   np.array([z[t] / cfg.z_crit]), fc)[0])
    F = np.asarray(rows, dtype=np.float32)
    return F.mean(0), F.std(0) + 1e-6


def _run_batch(policy, vcfg, fc, mu0, sd0, rng, B, steps, sigma, greedy):
    """Vectorized rollout of B parallel trajectories for `steps` steps (continuous, no reset).

    Returns (F[steps,B,D], a, mu, r) for training plus a trajectory dict of [steps,B] arrays.
    """
    m = fc.m
    x = np.zeros(B); z = np.zeros(B)
    Xh = np.zeros((B, m)); Ah = np.zeros((B, m)); Uh = np.zeros((B, m)); zc = np.zeros(B)
    Fs, As, Mus, Rs = [], [], [], []
    xs, us, vs, zs = [], [], [], []
    for _ in range(steps):
        F = (_feature_block(Xh, Ah, Uh, zc, fc) - mu0) / sd0
        mu = policy.mean(F)
        a = mu if greedy else np.clip(mu + sigma * rng.standard_normal(B), 0.0, 1.0)
        u, V, r, xn, zn = batch_step(vcfg, x, z, a, rng)
        Fs.append(F); As.append(a.copy()); Mus.append(mu.copy()); Rs.append(r)
        xs.append(xn.copy()); us.append(u); vs.append(V); zs.append(zn.copy())
        Xh = np.concatenate([Xh[:, 1:], xn[:, None]], axis=1)
        Ah = np.concatenate([Ah[:, 1:], a[:, None]], axis=1)
        Uh = np.concatenate([Uh[:, 1:], u[:, None].astype(float)], axis=1)
        zc = zn / vcfg.z_crit
        x, z = xn, zn
    traj = {"x": np.array(xs), "a": np.array(As), "u": np.array(us),
            "V": np.array(vs), "z": np.array(zs)}
    return np.array(Fs), np.array(As), np.array(Mus), np.array(Rs), traj


def train_agent(cfg, variant, fc: FeatCfg, hidden=64, iters=120, B=32,
                sigma=0.15, gamma=0.99, seed=0):
    rng = np.random.default_rng(seed)
    vcfg = cfg.for_variant(variant)
    mu0, sd0 = _norm_stats(cfg, variant, fc, rng)
    policy = Policy(fc.in_dim(), hidden, seed=seed)
    for _ in range(iters):
        F, A, Mu, R, _ = _run_batch(policy, vcfg, fc, mu0, sd0, rng, B, cfg.horizon, sigma, False)
        G = np.zeros_like(R); acc = np.zeros(R.shape[1])
        for t in range(R.shape[0] - 1, -1, -1):
            acc = R[t] + gamma * acc
            G[t] = acc
        adv = G.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        policy.policy_grad_step(F.reshape(-1, F.shape[2]), A.reshape(-1), Mu.reshape(-1), adv, sigma)
    return policy, mu0, sd0


def eval_agent(policy, cfg, variant, fc: FeatCfg, mu0, sd0, n_steps=40_000, B=8, seed=0):
    """Greedy continuous rollout; return (per-horizon return, violation rate, traj for absorber)."""
    rng = np.random.default_rng(seed)
    vcfg = cfg.for_variant(variant)
    steps = n_steps // B
    _, _, _, R, traj = _run_batch(policy, vcfg, fc, mu0, sd0, rng, B, steps, 0.0, True)
    per_horizon = float(R.sum(0).mean()) * (cfg.horizon / steps)
    vrate = float(np.mean(traj["V"]))
    flat = {k: np.concatenate([v[:, b] for b in range(B)]) for k, v in traj.items()}
    flat["u"] = flat["u"].astype(int)
    flat["V"] = flat["V"].astype(int)
    return per_horizon, vrate, flat
