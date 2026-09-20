"""Compact NumPy GRU binary classifier -- no deep-learning framework required.

A single-layer GRU reads the 48-hour observation window and the final hidden
state feeds a sigmoid head. Training is mini-batch Adam on binary
cross-entropy with backpropagation through time. This is a genuine recurrent
model (the hidden state carries information forward across hours), not a
feed-forward network on flattened sequences, so the repo's "deep learning"
path runs end-to-end on synthetic data with only NumPy as a dependency.

Reference equations (Cho et al., 2014):

    z_t = sigmoid(x_t W_z + h_{t-1} U_z + b_z)          (update gate)
    r_t = sigmoid(x_t W_r + h_{t-1} U_r + b_r)          (reset gate)
    h~_t = tanh(x_t W_h + (r_t * h_{t-1}) U_h + b_h)     (candidate)
    h_t = (1 - z_t) * h_{t-1} + z_t * h~_t
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class GRUClassifier:
    """Single-layer GRU + sigmoid head, trained with Adam."""

    def __init__(self, n_features: int, n_hidden: int = 32, seed: int = 0,
                 lr: float = 3e-3, epochs: int = 15, batch_size: int = 64,
                 l2: float = 1e-5, patience: int = 3, verbose: bool = False):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.seed = seed
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.patience = patience
        self.verbose = verbose
        self._init_params()

    # ------------------------------------------------------------------ init
    def _init_params(self) -> None:
        rng = np.random.default_rng(self.seed)
        F, H = self.n_features, self.n_hidden

        def w(shape, scale):
            return (rng.standard_normal(shape) * scale).astype(np.float64)

        sx = np.sqrt(1.0 / F)
        sh = np.sqrt(1.0 / H)
        self.params = {
            "Wz": w((F, H), sx), "Uz": w((H, H), sh), "bz": np.zeros(H),
            "Wr": w((F, H), sx), "Ur": w((H, H), sh), "br": np.zeros(H),
            "Wh": w((F, H), sx), "Uh": w((H, H), sh), "bh": np.zeros(H),
            "wo": w((H, 1), sh), "bo": np.zeros(1),
        }

    # -------------------------------------------------------------- forward
    def _forward(self, X: np.ndarray):
        """Return (logits, cache). X: (B, T, F)."""
        P = self.params
        B, T, _ = X.shape
        H = self.n_hidden
        h = np.zeros((B, H))
        steps = []
        for t in range(T):
            x = X[:, t, :]
            z = _sigmoid(x @ P["Wz"] + h @ P["Uz"] + P["bz"])
            r = _sigmoid(x @ P["Wr"] + h @ P["Ur"] + P["br"])
            a = r * h
            hh = np.tanh(x @ P["Wh"] + a @ P["Uh"] + P["bh"])
            h_new = (1.0 - z) * h + z * hh
            steps.append((x, h, z, r, a, hh, h_new))
            h = h_new
        logits = (h @ P["wo"] + P["bo"]).ravel()
        return logits, (steps, h)

    # ------------------------------------------------------------- backward
    def _loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        """Mean binary cross-entropy + L2, and its gradients (BPTT)."""
        P = self.params
        B = X.shape[0]
        logits, (steps, h_last) = self._forward(X)
        p = _sigmoid(logits)
        eps = 1e-12
        loss = float(-np.mean(y * np.log(p + eps)
                              + (1 - y) * np.log(1 - p + eps)))
        loss += 0.5 * self.l2 * sum(float((v * v).sum())
                                    for k, v in P.items() if k[0] in "WU")

        dlogit = ((p - y) / B)[:, None]          # (B, 1)
        grads = {k: np.zeros_like(v) for k, v in P.items()}
        grads["wo"] = h_last.T @ dlogit + self.l2 * P["wo"]
        grads["bo"] = dlogit.sum(axis=0)

        dh = dlogit @ P["wo"].T                   # into final hidden state
        for x, hp, z, r, a, hh, _ in reversed(steps):
            dhh = dh * z
            dz = dh * (hh - hp)
            dhp = dh * (1.0 - z)

            dhh_pre = dhh * (1.0 - hh ** 2)
            grads["Wh"] += x.T @ dhh_pre
            grads["Uh"] += a.T @ dhh_pre
            grads["bh"] += dhh_pre.sum(axis=0)

            da = dhh_pre @ P["Uh"].T
            dr = da * hp
            dhp = dhp + da * r
            dr_pre = dr * r * (1.0 - r)
            grads["Wr"] += x.T @ dr_pre
            grads["Ur"] += hp.T @ dr_pre
            grads["br"] += dr_pre.sum(axis=0)
            dhp = dhp + dr_pre @ P["Ur"].T

            dz_pre = dz * z * (1.0 - z)
            grads["Wz"] += x.T @ dz_pre
            grads["Uz"] += hp.T @ dz_pre
            grads["bz"] += dz_pre.sum(axis=0)
            dhp = dhp + dz_pre @ P["Uz"].T

            dh = dhp

        for k in grads:
            if k[0] in "WU":
                grads[k] += self.l2 * P[k]
        return loss, grads

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.seed + 1)
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0
        best = None
        best_params = None
        bad = 0
        n = X.shape[0]

        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                loss, grads = self._loss_and_grads(X[idx], y[idx])
                step += 1
                for k in self.params:
                    g = np.clip(grads[k], -5.0, 5.0)
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v[k] = b2 * v[k] + (1 - b2) * g * g
                    mh = m[k] / (1 - b1 ** step)
                    vh = v[k] / (1 - b2 ** step)
                    self.params[k] -= self.lr * mh / (np.sqrt(vh) + eps)

            if X_val is not None:
                vloss, _ = self._loss_and_grads(
                    np.asarray(X_val, dtype=np.float64),
                    np.asarray(y_val, dtype=np.float64))
                if self.verbose:
                    print(f"epoch {epoch + 1}/{self.epochs} "
                          f"train_loss={loss:.4f} val_loss={vloss:.4f}")
                if best is None or vloss < best - 1e-4:
                    best = vloss
                    best_params = {k: v.copy()
                                   for k, v in self.params.items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        if self.verbose:
                            print(f"early stopping at epoch {epoch + 1}")
                        break
            elif self.verbose:
                print(f"epoch {epoch + 1}/{self.epochs} train_loss={loss:.4f}")

        if best_params is not None:
            self.params = best_params
        return self

    # -------------------------------------------------------------- predict
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        logits, _ = self._forward(X)
        p = _sigmoid(logits)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
