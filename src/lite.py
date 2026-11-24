"""The light engine, for small servers (Render's free plan has 512 MB).

TensorFlow alone needs more memory than that, so on such a server:

- the trained CNN runs in NumPy: the same weights, exported from
  models/digit_cnn.h5 by scripts/export_lite.py (batch normalisation folded
  into the convolutions), giving the same probabilities as Keras;
- retraining keeps the convolution layers fixed and retrains the last layer
  (64 -> 10) on the original data plus the uploads, starting from the
  current model's weights. This is transfer learning: the trained network is
  the feature extractor, and only its last layer learns the new voices.

The original training clips are stored as their 64-number embeddings
(models/lite/embeddings.npz), so retraining doesn't have to reprocess
thousands of recordings on a slow CPU.
"""
import numpy as np
from scipy.optimize import minimize

from src.config import MODELS_DIR

LITE_DIR = MODELS_DIR / "lite"
BASE_LITE_PATH = LITE_DIR / "digit_cnn.npz"
BASE_LITE_INFO = LITE_DIR / "digit_cnn_info.json"
EMBEDDINGS_PATH = LITE_DIR / "embeddings.npz"
LAYERS = ["conv1", "conv2", "conv3", "conv4", "dense1", "head"]


def _conv3x3_relu(x, w, b):
    """3x3 convolution with 'same' padding, then ReLU. x: (N, H, W, C)."""
    n, h, wd, c = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(1, 2))   # N,H,W,C,3,3
    cols = win.transpose(0, 1, 2, 4, 5, 3).reshape(n, h, wd, 9 * c)
    return np.maximum(cols @ w.reshape(9 * c, -1) + b, 0)


def _maxpool2(x):
    n, h, w, c = x.shape
    h2, w2 = h // 2, w // 2
    return x[:, :h2 * 2, :w2 * 2].reshape(n, h2, 2, w2, 2, c).max(axis=(2, 4))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class NumpyCNN:
    def __init__(self, path=BASE_LITE_PATH, weights=None):
        z = dict(np.load(path)) if weights is None else weights
        self.w = {k: z[k].astype(np.float32) for k in z}

    def embed(self, X, batch=64):
        """The 64 numbers the last layer sees (after the first dense layer)."""
        out = []
        for i in range(0, len(X), batch):
            x = np.asarray(X[i:i + batch], np.float32)
            for k, name in enumerate(["conv1", "conv2", "conv3", "conv4"]):
                x = _conv3x3_relu(x, self.w[f"{name}_w"], self.w[f"{name}_b"])
                if k < 3:
                    x = _maxpool2(x)
            x = x.mean(axis=(1, 2))
            out.append(np.maximum(x @ self.w["dense1_w"] + self.w["dense1_b"], 0))
        return np.concatenate(out) if out else np.zeros((0, 64), np.float32)

    def head_proba(self, E):
        return softmax(np.asarray(E, np.float32) @ self.w["head_w"] + self.w["head_b"])

    def predict(self, X):
        return self.head_proba(self.embed(X))

    def with_head(self, W, b):
        weights = dict(self.w)
        weights["head_w"], weights["head_b"] = W.astype(np.float32), b.astype(np.float32)
        return NumpyCNN(weights=weights)

    def save(self, path):
        np.savez_compressed(path, **self.w)


def export_from_keras(model):
    """Keras model -> dict of NumPy weights, batch norm folded into the convs."""
    convs = [l for l in model.layers if l.__class__.__name__ == "Conv2D"]
    norms = [l for l in model.layers if l.__class__.__name__ == "BatchNormalization"]
    denses = [l for l in model.layers if l.__class__.__name__ == "Dense"]
    out = {}
    for i, (conv, bn) in enumerate(zip(convs, norms), start=1):
        k = conv.get_weights()[0]
        gamma, beta, mean, var = bn.get_weights()
        scale = gamma / np.sqrt(var + bn.epsilon)
        out[f"conv{i}_w"] = (k * scale).astype(np.float32)
        out[f"conv{i}_b"] = (beta - mean * scale).astype(np.float32)
    out["dense1_w"], out["dense1_b"] = [a.astype(np.float32) for a in denses[0].get_weights()]
    out["head_w"], out["head_b"] = [a.astype(np.float32) for a in denses[1].get_weights()]
    return out


def embeddings():
    return dict(np.load(EMBEDDINGS_PATH))


def fit_head(E, y, sample_weight, W0, b0, l2=1e-3, max_iter=150, on_step=None, every=10):
    """Weighted softmax regression on embeddings, started from (W0, b0).
    Returns the new weights and a history like Keras' (one point per `every` steps)."""
    E = np.asarray(E, np.float64)
    n_in, n_out = W0.shape
    Y = np.eye(n_out)[y]
    w = np.asarray(sample_weight, np.float64)
    w = w / w.sum()

    def loss_grad(theta):
        W = theta[:n_in * n_out].reshape(n_in, n_out)
        b = theta[n_in * n_out:]
        P = softmax(E @ W + b)
        loss = -np.sum(w * np.log(np.clip((P * Y).sum(1), 1e-12, None))) + l2 * np.sum(W ** 2)
        G = (P - Y) * w[:, None]
        return loss, np.concatenate([(E.T @ G + 2 * l2 * W).ravel(), G.sum(0)])

    state = {"step": 0}

    def callback(theta):
        state["step"] += 1
        if on_step and state["step"] % every == 0:
            on_step(state["step"] // every, theta[:n_in * n_out].reshape(n_in, n_out), theta[n_in * n_out:])

    theta0 = np.concatenate([W0.ravel(), b0]).astype(np.float64)
    res = minimize(loss_grad, theta0, jac=True, method="L-BFGS-B", callback=callback,
                   options={"maxiter": max_iter})
    W = res.x[:n_in * n_out].reshape(n_in, n_out)
    return W.astype(np.float32), res.x[n_in * n_out:].astype(np.float32)
