import numpy as np
import pytest

from deixis.gaze import model
from deixis.gaze.features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    LEFT_EYE,
    RIGHT_EYE,
    eye_ratios,
    head_pose,
)
from deixis.gaze.filter import OneEuro
from deixis.gaze.history import GazeHistory, fixations, primary

H = FEATURE_NAMES.index("r_h")
V = FEATURE_NAMES.index("r_v")


def synthetic(targets, rng, per_target=25, noise=0.004):
    rows, ys, groups = [], [], []
    for g, (x, y) in enumerate(targets):
        for _ in range(per_target):
            f = rng.normal(0, 0.01, FEATURE_DIM)
            f[H] = f[FEATURE_NAMES.index("l_h")] = 0.35 + 0.3 * x / 1440
            f[V] = f[FEATURE_NAMES.index("l_v")] = 0.4 + 0.2 * y / 900
            f[[H, V]] += rng.normal(0, noise, 2)
            rows.append(f)
            ys.append((x, y))
            groups.append(g)
    return np.array(rows), np.array(ys, dtype=float), np.array(groups)


def test_model_recovers_mapping_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    grid = [(x, y) for x in (100, 720, 1340) for y in (80, 450, 820)]
    grid += [(400, 260), (1040, 260), (400, 640), (1040, 640)]
    feats, targets, groups = synthetic(grid, rng)
    fitted = model.fit(feats, targets, groups)
    assert fitted.cv_error_px < 80

    probe, truth, _ = synthetic([(600, 300)], rng, per_target=40)
    predicted = fitted.predict(probe).mean(axis=0)
    assert np.linalg.norm(predicted - truth[0]) < 60

    path = tmp_path / "cal.npz"
    fitted.save(path)
    loaded = model.GazeModel.load(path)
    assert loaded is not None
    np.testing.assert_allclose(loaded.predict(probe[0]), fitted.predict(probe[0]))


def test_model_needs_enough_targets():
    rng = np.random.default_rng(1)
    feats, targets, groups = synthetic([(0, 0), (100, 100), (200, 0)], rng)
    with pytest.raises(ValueError):
        model.fit(feats, targets, groups)


def test_both_eyes_horizontal_ratios_move_together():
    points = np.zeros((478, 2))
    for eye, x0 in ((RIGHT_EYE, 100.0), (LEFT_EYE, 200.0)):
        left, right, upper, lower, _ = eye
        points[left], points[right] = (x0, 50), (x0 + 40, 50)
        points[upper], points[lower] = (x0 + 20, 40), (x0 + 20, 60)

    def ratios(shift):
        points[RIGHT_EYE[4]] = (120 + shift, 50)
        points[LEFT_EYE[4]] = (220 + shift, 50)
        return eye_ratios(points, RIGHT_EYE)[0], eye_ratios(points, LEFT_EYE)[0]

    (r0, l0), (r1, l1) = ratios(0), ratios(8)
    assert r0 == pytest.approx(l0)
    assert r1 > r0 and l1 > l0


def test_head_pose_of_identity_is_zero():
    m = np.eye(4)
    m[:3, 3] = (1.0, 2.0, -50.0)
    yaw, pitch, roll, tx, ty, tz = head_pose(m)
    assert (yaw, pitch, roll) == pytest.approx((0, 0, 0))
    assert (tx, ty, tz) == (1.0, 2.0, -50.0)


def test_fixations_split_on_saccade_and_primary_is_longest():
    t = np.arange(0, 1.5, 1 / 30)
    xs = np.where(t < 0.4, 200.0, 900.0)
    samples = np.column_stack([t, xs, np.full_like(t, 300.0)])
    found = fixations(samples)
    assert len(found) == 2
    assert found[0].x == pytest.approx(200)
    assert primary(found).x == pytest.approx(900)


def test_fixations_drop_short_glances():
    t = np.arange(0, 1.0, 1 / 30)
    xs = np.full_like(t, 500.0)
    xs[10:12] = 1200.0
    found = fixations(np.column_stack([t, xs, np.full_like(t, 100.0)]))
    assert len(found) == 1
    assert found[0].x == pytest.approx(500)


def test_history_window_and_expiry():
    h = GazeHistory(seconds=1.0)
    for i in range(30):
        h.add(i * 0.1, float(i), 0.0)
    assert h.window(0, 100).shape[0] <= 11
    assert h.latest()[1] == 29.0


def test_one_euro_smooths_jitter_but_follows_jumps():
    f = OneEuro()
    rng = np.random.default_rng(2)
    out = [f(i / 30, 500 + rng.normal(0, 30), 500) for i in range(60)]
    assert np.std([o[0] for o in out[30:]]) < 20
    for i in range(60, 90):
        x, _ = f(i / 30, 1200, 500)
    assert x == pytest.approx(1200, abs=40)
