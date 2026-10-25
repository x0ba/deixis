"""Maps gaze features to screen points with ridge regression.

Samples collected while looking at one calibration target are strongly
correlated, so the feature set and regularization strength are picked by
leave-one-target-out cross-validation rather than per-sample. Which features
help (eyelid blendshapes, head translation) differs between people, cameras
and lighting, so each calibration tries a few sets and keeps the best.

People hold their head still during calibration, so head pose barely varies
and standardizing by its calibration spread would turn a later lean of a few
centimeters into a huge input. Each pose feature's scale is floored at a
typical posture change instead.

Samples in group -1 (from following a moving dot) are only ever used for
training: their targets are less exact, so errors are measured on fixed dots.

A handful of dots is too few to fit a new model, but enough to adapt an
existing one to someone else: `adapt` fits an affine correction (scale, skew
and offset) of its predictions, pulled toward no change.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from peek.gaze.features import FEATURE_DIM, FEATURE_NAMES, FEATURES_VERSION, LOOK_BLENDSHAPES

ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
ADAPT_ALPHA = 0.05  # in screen fractions, so it doesn't depend on screen size
MIN_ADAPT_DOTS = 4
MIN_STD = {"yaw": 0.05, "pitch": 0.05, "roll": 0.05, "tx": 1.5, "ty": 1.5, "tz": 3.0}
_MIN_STD = np.array([MIN_STD.get(name, 0.0) for name in FEATURE_NAMES])

_TRANSLATION = ("tx", "ty", "tz")
FEATURE_SETS = {
    "all": (),
    "no_blendshapes": LOOK_BLENDSHAPES,
    "no_translation": _TRANSLATION,
    "eyes_and_angles": (*LOOK_BLENDSHAPES, *_TRANSLATION),
}

_H = (FEATURE_NAMES.index("r_h"), FEATURE_NAMES.index("l_h"))
_V = (FEATURE_NAMES.index("r_v"), FEATURE_NAMES.index("l_v"))
_YAW = FEATURE_NAMES.index("yaw")
_PITCH = FEATURE_NAMES.index("pitch")


def feature_mask(name: str) -> np.ndarray:
    dropped = FEATURE_SETS[name]
    return np.array([0.0 if f in dropped else 1.0 for f in FEATURE_NAMES])


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
    mask: np.ndarray = field(default_factory=lambda: np.ones(FEATURE_DIM))
    feature_set: str = "all"
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    checked_error_px: float | None = None  # measured on fresh dots after fitting

    @property
    def error_px(self) -> float:
        """Typical distance between predicted and true gaze."""
        return self.checked_error_px if self.checked_error_px is not None else self.cv_error_px

    def predict(self, features: np.ndarray) -> np.ndarray:
        single = features.ndim == 1
        z = (np.atleast_2d(features) - self.mean) / self.std * self.mask
        out = expand(z) @ self.weights + self.offset
        return out[0] if single else out

    def recentered(self, features: np.ndarray, target: tuple[float, float]) -> "GazeModel":
        """A copy whose offset puts the median prediction for `features` on `target`."""
        shift = np.asarray(target, dtype=float) - np.median(self.predict(features), axis=0)
        return replace(self, offset=self.offset + shift)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            std=self.std,
            weights=self.weights,
            alpha=self.alpha,
            cv_error_px=self.cv_error_px,
            mask=self.mask,
            feature_set=self.feature_set,
            offset=self.offset,
            checked_error_px=np.nan if self.checked_error_px is None else self.checked_error_px,
            version=FEATURES_VERSION,
        )

    @classmethod
    def load(cls, path: Path) -> "GazeModel | None":
        if not path.exists():
            return None
        data = np.load(path)
        if "version" not in data or int(data["version"]) != FEATURES_VERSION:
            return None
        if data["mean"].shape != (FEATURE_DIM,):
            return None
        checked = float(data["checked_error_px"]) if "checked_error_px" in data else np.nan
        return cls(
            mean=data["mean"],
            std=data["std"],
            weights=data["weights"],
            alpha=float(data["alpha"]),
            cv_error_px=float(data["cv_error_px"]),
            mask=data["mask"],
            feature_set=str(data["feature_set"]),
            offset=data["offset"],
            checked_error_px=checked if np.isfinite(checked) else None,
        )


def _cv_error(
    x: np.ndarray, targets: np.ndarray, groups: np.ndarray, visits: np.ndarray, alpha: float
) -> float:
    errors = []
    for g in np.unique(groups[groups >= 0]):
        train, test = groups != g, groups == g
        w = _solve(x[train], targets[train], alpha)
        predicted = x[test] @ w
        for v in np.unique(visits[test]):
            here = visits[test] == v
            errors.append(np.linalg.norm(predicted[here].mean(axis=0) - targets[test][here][0]))
    return float(np.mean(errors))


def fit(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    visits: np.ndarray | None = None,
) -> GazeModel:
    """Fit on (n, FEATURE_DIM) features and (n, 2) screen targets.

    `groups` identifies the screen position of each sample and is held out
    together (-1 marks training-only samples); `visits` identifies each
    separate look at a position (defaults to `groups`) and is what the error
    is averaged over.
    """
    visits = groups if visits is None else visits
    if not (len(features) == len(targets) == len(groups) == len(visits)):
        raise ValueError("features, targets, groups and visits must align")
    if len(np.unique(groups[groups >= 0])) < 4:
        raise ValueError("need at least 4 calibration targets")
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-9] = 1.0
    std = np.maximum(std, _MIN_STD)
    z = (features - mean) / std

    best = ("all", ALPHAS[0], float("inf"))
    for name in FEATURE_SETS:
        x = expand(z * feature_mask(name))
        for alpha in ALPHAS:
            err = _cv_error(x, targets, groups, visits, alpha)
            if err < best[2]:
                best = (name, alpha, err)

    name, alpha, err = best
    mask = feature_mask(name)
    weights = _solve(expand(z * mask), targets, alpha)
    return GazeModel(
        mean=mean,
        std=std,
        weights=weights,
        alpha=alpha,
        cv_error_px=err,
        mask=mask,
        feature_set=name,
    )


def _affine(predicted: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """The (3, 2) map taking [x, y, 1] predictions to targets, both in screen
    fractions, with its linear part pulled toward the identity."""
    p = np.hstack([predicted, np.ones((len(predicted), 1))])
    identity = np.vstack([np.eye(2), np.zeros((1, 2))])
    penalty = ADAPT_ALPHA * np.diag([1.0, 1.0, 0.0])
    return identity + np.linalg.solve(p.T @ p + penalty, p.T @ (targets - p @ identity))


def adapt(
    model: GazeModel,
    features: np.ndarray,
    targets: np.ndarray,
    visits: np.ndarray,
    screen: tuple[float, float],
) -> GazeModel:
    """`model` adapted to whoever looked at `targets`, one dot per visit.

    Whichever of `model` with an affine correction and `model` unchanged is
    more accurate on held-out dots, with that error as `checked_error_px`.
    """
    dots = np.unique(visits)
    if len(dots) < MIN_ADAPT_DOTS:
        raise ValueError(f"need at least {MIN_ADAPT_DOTS} dots")
    scale = np.asarray(screen, dtype=float)
    predicted = (
        np.array([np.median(model.predict(features[visits == v]), axis=0) for v in dots]) / scale
    )
    goals = np.array([targets[visits == v][0] for v in dots]) / scale

    held_out = []
    for i in range(len(dots)):
        a = _affine(np.delete(predicted, i, axis=0), np.delete(goals, i, axis=0))
        held_out.append(np.linalg.norm((np.append(predicted[i], 1.0) @ a - goals[i]) * scale))
    corrected = float(np.mean(held_out))
    unchanged = float(np.mean(np.linalg.norm((predicted - goals) * scale, axis=1)))
    if unchanged <= corrected:
        return replace(model, checked_error_px=unchanged)

    a = _affine(predicted, goals)
    linear = a[:2] * scale[None, :] / scale[:, None]
    return replace(
        model,
        weights=model.weights @ linear,
        offset=model.offset @ linear + a[2] * scale,
        checked_error_px=corrected,
    )
