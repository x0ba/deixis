"""Maps gaze features to screen points with ridge regression.

Samples collected while looking at one calibration target are strongly
correlated, so the regularization strength is picked by leave-one-target-out
cross-validation rather than per-sample.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deixis.gaze.features import FEATURE_DIM, FEATURE_NAMES

ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)

_H = (FEATURE_NAMES.index("r_h"), FEATURE_NAMES.index("l_h"))
_V = (FEATURE_NAMES.index("r_v"), FEATURE_NAMES.index("l_v"))
_YAW = FEATURE_NAMES.index("yaw")
_PITCH = FEATURE_NAMES.index("pitch")


def expand(z: np.ndarray) -> np.ndarray:
    """Standardized features plus a few interaction terms. `z` is (n, FEATURE_DIM)."""
    h = z[:, _H].mean(axis=1, keepdims=True)
    v = z[:, _V].mean(axis=1, keepdims=True)
    yaw = z[:, [_YAW]]
    pitch = z[:, [_PITCH]]
    return np.hstack([z, h * h, v * v, h * v, h * yaw, v * pitch, np.ones((len(z), 1))])


def _solve(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = alpha * np.eye(x.shape[1])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


@dataclass
class GazeModel:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    alpha: float
    cv_error_px: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        single = features.ndim == 1
        z = (np.atleast_2d(features) - self.mean) / self.std
        out = expand(z) @ self.weights
        return out[0] if single else out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            std=self.std,
            weights=self.weights,
            alpha=self.alpha,
            cv_error_px=self.cv_error_px,
        )

    @classmethod
    def load(cls, path: Path) -> "GazeModel | None":
        if not path.exists():
            return None
        data = np.load(path)
        if data["mean"].shape != (FEATURE_DIM,):
            return None
        return cls(
            mean=data["mean"],
            std=data["std"],
            weights=data["weights"],
            alpha=float(data["alpha"]),
            cv_error_px=float(data["cv_error_px"]),
        )


def fit(features: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> GazeModel:
    """Fit on (n, FEATURE_DIM) features, (n, 2) screen targets, and (n,) target ids."""
    if len(features) != len(targets) or len(features) != len(groups):
        raise ValueError("features, targets and groups must align")
    unique = np.unique(groups)
    if len(unique) < 4:
        raise ValueError("need at least 4 calibration targets")
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-9] = 1.0
    x = expand((features - mean) / std)

    best_alpha, best_err = ALPHAS[0], float("inf")
    for alpha in ALPHAS:
        errors = []
        for g in unique:
            train, test = groups != g, groups == g
            w = _solve(x[train], targets[train], alpha)
            predicted = (x[test] @ w).mean(axis=0)
            errors.append(np.linalg.norm(predicted - targets[test][0]))
        err = float(np.mean(errors))
        if err < best_err:
            best_alpha, best_err = alpha, err

    weights = _solve(x, targets, best_alpha)
    return GazeModel(mean=mean, std=std, weights=weights, alpha=best_alpha, cv_error_px=best_err)
