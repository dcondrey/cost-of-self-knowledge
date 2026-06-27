"""Closed-loop absorber (ASDS, EXPERIMENT.md Section 3/4).

A pure-numpy windowed MLP that predicts B's next intervention u_{t+1} from A's observable
history O_t = [x, a, u over a window m]. Its converged held-out cross-entropy (in bits) is an
upper bound on H(u_{t+1}|O_t) that tightens with capacity; phi_effective = that / H(u). This
is the model that stands in for "the best A could do" when measuring non-absorption.

No autodiff dependency: explicit forward/backward + Adam. Scales by widening `hidden`.
"""
import numpy as np


def featurize(traj, m, K):
    """Windowed observable features at each t -> integer target u_{t+1}.

    Features: [x_{t-m+1..t}, a_{t-m+1..t}, one-hot(u)_{t-m+1..t}], standardized columns.
    """
    x, a, u = traj["x"], traj["a"], traj["u"]
    n = len(u)
    onehot = np.eye(K, dtype=np.float32)[u]
    rows, tgt = [], []
    for t in range(m - 1, n - 1):
        win_x = x[t - m + 1:t + 1]
        win_a = a[t - m + 1:t + 1]
        win_u = onehot[t - m + 1:t + 1].reshape(-1)
        rows.append(np.concatenate([win_x, win_a, win_u]))
        tgt.append(u[t + 1])
    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(tgt, dtype=np.int64)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return (X - mu) / sd, y


class MLP:
    """Feedforward classifier, ReLU hidden layers, softmax output, Adam."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for nin, nout in zip(sizes[:-1], sizes[1:]):
            self.W.append((rng.standard_normal((nin, nout)) * np.sqrt(2.0 / nin)).astype(np.float32))
            self.b.append(np.zeros(nout, dtype=np.float32))
        self._init_adam()

    def n_params(self):
        return sum(w.size for w in self.W) + sum(bb.size for bb in self.b)

    def _init_adam(self):
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(bb) for bb in self.b]
        self.vb = [np.zeros_like(bb) for bb in self.b]
        self.t = 0

    def forward(self, X):
        acts = [X]
        h = X
        for i in range(len(self.W) - 1):
            h = np.maximum(0.0, h @ self.W[i] + self.b[i])
            acts.append(h)
        logits = h @ self.W[-1] + self.b[-1]
        return logits, acts

    @staticmethod
    def _softmax(logits):
        z = logits - logits.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def loss_bits(self, X, y):
        """Mean cross-entropy in bits (= empirical H(u|features) upper bound)."""
        logits, _ = self.forward(X)
        p = self._softmax(logits)
        return float(-np.mean(np.log2(p[np.arange(len(y)), y] + 1e-12)))

    def _step(self, X, y, lr):
        logits, acts = self.forward(X)
        p = self._softmax(logits)
        grad = p
        grad[np.arange(len(y)), y] -= 1.0
        grad /= len(y)
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ grad
            gb[i] = grad.sum(0)
            if i > 0:
                grad = (grad @ self.W[i].T) * (acts[i] > 0)
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.W)):
            for g, mlist, vlist, param in ((gW[i], self.mW, self.vW, self.W),
                                           (gb[i], self.mb, self.vb, self.b)):
                mlist[i] = b1 * mlist[i] + (1 - b1) * g
                vlist[i] = b2 * vlist[i] + (1 - b2) * g * g
                mhat = mlist[i] / (1 - b1 ** self.t)
                vhat = vlist[i] / (1 - b2 ** self.t)
                param[i] -= lr * mhat / (np.sqrt(vhat) + eps)

    def fit(self, X, y, Xv, yv, epochs=60, batch=256, lr=1e-3, patience=8, seed=0):
        rng = np.random.default_rng(seed)
        best, best_state, wait = float("inf"), None, 0
        n = len(y)
        for _ in range(epochs):
            for idx in np.array_split(rng.permutation(n), max(1, n // batch)):
                self._step(X[idx], y[idx], lr)
            vloss = self.loss_bits(Xv, yv)
            if vloss < best - 1e-4:
                best, wait = vloss, 0
                best_state = ([w.copy() for w in self.W], [bb.copy() for bb in self.b])
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_state is not None:
            self.W, self.b = best_state
        return best


def marginal_entropy_bits(u, K):
    counts = np.bincount(u, minlength=K).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def measure_phi_effective(traj, m, hidden, K, seed=0, val_frac=0.2):
    """Train the absorber on a trajectory; return (phi_effective, val_ce_bits, n_params).

    phi_effective = converged held-out CE(bits) / H(u). Train/val split is contiguous (no
    leakage across the time series).
    """
    X, y = featurize(traj, m, K)
    n = len(y)
    cut = int(n * (1 - val_frac))
    Xtr, ytr, Xv, yv = X[:cut], y[:cut], X[cut:], y[cut:]
    net = MLP([X.shape[1], *hidden, K], seed=seed)
    val_ce = net.fit(Xtr, ytr, Xv, yv, seed=seed)
    h_marg = marginal_entropy_bits(y, K)
    phi = val_ce / h_marg if h_marg > 0 else 0.0
    return phi, val_ce, net.n_params()
