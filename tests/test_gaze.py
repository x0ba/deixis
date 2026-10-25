from dataclasses import replace

import numpy as np
import pytest

from peek.gaze import drift, model
from peek.gaze.features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    LEFT_EYE,
    LOOK_BLENDSHAPES,
    RIGHT_EYE,
    eye_ratios,
    head_pose,
)
from peek.gaze.filter import OneEuro
from peek.gaze.history import GazeHistory, fixations, primary
from peek.gaze.tracker import CameraGaze
from peek.screen.ax import Rect
from peek.ui.calibration import PURSUIT_PX_S, Pursuit, _pick, check_error

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
        left, right, _ = eye
        points[left], points[right] = (x0, 50), (x0 + 40, 50)

    def ratios(shift):
        points[RIGHT_EYE[2]] = (120 + shift, 50)
        points[LEFT_EYE[2]] = (220 + shift, 50)
        return eye_ratios(points, RIGHT_EYE)[0], eye_ratios(points, LEFT_EYE)[0]

    (r0, l0), (r1, l1) = ratios(0), ratios(8)
    assert r0 == pytest.approx(l0)
    assert r1 > r0 and l1 > l0


def test_vertical_ratio_is_measured_from_the_corner_line():
    points = np.zeros((478, 2))
    left, right, iris = RIGHT_EYE
    points[left], points[right] = (100, 50), (140, 50)
    points[iris] = (120, 50)
    assert eye_ratios(points, RIGHT_EYE)[1] == pytest.approx(0)
    points[iris] = (120, 54)
    assert eye_ratios(points, RIGHT_EYE)[1] == pytest.approx(0.1)

    angle = np.radians(15)
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rolled = points.copy()
    for i in RIGHT_EYE:
        rolled[i] = (points[i] - points[left]) @ turn.T + points[left]
    np.testing.assert_allclose(eye_ratios(rolled, RIGHT_EYE), eye_ratios(points, RIGHT_EYE))


def test_model_ignores_posture_changes_it_never_saw():
    rng = np.random.default_rng(3)
    grid = [(x, y) for x in (100, 720, 1340) for y in (80, 450, 820)]
    feats, targets, groups = synthetic(grid, rng)
    tz = FEATURE_NAMES.index("tz")
    feats[:, tz] = -50 + 0.4 * targets[:, 1] / 900 + rng.normal(0, 0.02, len(feats))
    fitted = model.fit(feats, targets, groups)

    probe, _, _ = synthetic([(600, 300)], rng, per_target=40)
    probe[:, tz] = -50 + 0.4 * 300 / 900
    still = fitted.predict(probe).mean(axis=0)
    probe[:, tz] += 5
    leaned = fitted.predict(probe).mean(axis=0)
    assert np.linalg.norm(leaned - still) < 60


def test_calibration_from_older_features_is_not_loaded(tmp_path):
    rng = np.random.default_rng(4)
    grid = [(x, y) for x in (100, 720, 1340) for y in (80, 450, 820)]
    fitted = model.fit(*synthetic(grid, rng))
    path = tmp_path / "cal.npz"
    np.savez(
        path, mean=fitted.mean, std=fitted.std, weights=fitted.weights, alpha=1.0, cv_error_px=1.0
    )
    assert model.GazeModel.load(path) is None


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


GRID_3X3 = [(x, y) for x in (100, 720, 1340) for y in (80, 450, 820)]


def test_fit_keeps_the_features_that_carry_the_signal():
    rng = np.random.default_rng(5)
    feats, targets, groups = synthetic(GRID_3X3, rng)
    for name in ("r_h", "l_h"):
        feats[:, FEATURE_NAMES.index(name)] = rng.normal(0.5, 0.01, len(feats))
    look_out = FEATURE_NAMES.index("eyeLookOutLeft")
    feats[:, look_out] = targets[:, 0] / 1440 + rng.normal(0, 0.005, len(feats))
    fitted = model.fit(feats, targets, groups)
    assert "eyeLookOutLeft" not in model.FEATURE_SETS[fitted.feature_set]
    assert fitted.cv_error_px < 80


def test_dropped_features_do_not_affect_predictions(tmp_path):
    rng = np.random.default_rng(6)
    fitted = model.fit(*synthetic(GRID_3X3, rng))
    fitted.mask = model.feature_mask("eyes_and_angles")
    probe, _, _ = synthetic([(600, 300)], rng, per_target=1)
    before = fitted.predict(probe[0])
    for name in LOOK_BLENDSHAPES + ("tx", "ty", "tz"):
        probe[0, FEATURE_NAMES.index(name)] += 5.0
    np.testing.assert_allclose(fitted.predict(probe[0]), before)

    path = tmp_path / "cal.npz"
    fitted.save(path)
    loaded = model.GazeModel.load(path)
    np.testing.assert_allclose(loaded.mask, fitted.mask)
    assert loaded.feature_set == fitted.feature_set


def test_recentering_moves_the_median_prediction_onto_the_target(tmp_path):
    rng = np.random.default_rng(7)
    fitted = model.fit(*synthetic(GRID_3X3, rng))
    probe, _, _ = synthetic([(600, 300)], rng, per_target=30)
    probe[:, H] += 0.02
    shifted = fitted.recentered(probe, (600.0, 300.0))
    np.testing.assert_allclose(np.median(shifted.predict(probe), axis=0), (600, 300), atol=1e-6)
    assert np.allclose(fitted.offset, 0)

    path = tmp_path / "cal.npz"
    shifted.save(path)
    np.testing.assert_allclose(model.GazeModel.load(path).offset, shifted.offset)


def test_cv_error_is_per_visit_not_per_position():
    rng = np.random.default_rng(8)
    feats, targets, groups = synthetic(GRID_3X3 * 2, rng)
    groups = groups % len(GRID_3X3)
    visits = np.arange(len(groups)) // 25
    half = len(feats) // 2
    feats[:half, H] += 0.03
    feats[half:, H] -= 0.03
    pooled = model.fit(feats, targets, groups)
    split = model.fit(feats, targets, groups, visits)
    assert split.cv_error_px > pooled.cv_error_px


def test_training_only_samples_never_count_as_error():
    rng = np.random.default_rng(4)
    grid = [(x, y) for x in (100, 720, 1340) for y in (80, 450, 820)]
    feats, targets, groups = synthetic(grid, rng)
    extra, extra_targets, _ = synthetic([(400, 260), (1040, 640)], rng)
    # Wildly wrong targets on training-only rows would wreck the error if they were tested.
    feats = np.vstack([feats, extra])
    targets = np.vstack([targets, extra_targets + 5000])
    groups = np.concatenate([groups, np.full(len(extra), -1)])
    fitted = model.fit(feats, targets, groups, groups.copy())
    assert np.isfinite(fitted.cv_error_px)


def test_checked_error_round_trips_and_wins(tmp_path):
    rng = np.random.default_rng(5)
    grid = [(x, y) for x in (100, 720, 1340) for y in (80, 450, 820)]
    fitted = model.fit(*synthetic(grid, rng))
    assert fitted.error_px == fitted.cv_error_px
    path = tmp_path / "cal.npz"
    fitted.save(path)
    assert model.GazeModel.load(path).checked_error_px is None
    checked = replace(fitted, checked_error_px=123.0)
    checked.save(path)
    assert model.GazeModel.load(path).error_px == 123.0


def test_drift_shift_moves_part_way_and_ignores_big_misses():
    predicted = np.tile([500.0, 300.0], (10, 1))
    step = drift.shift(predicted, (540.0, 320.0), error_px=100.0)
    np.testing.assert_allclose(step, [40 * drift.POINT_GAIN, 20 * drift.POINT_GAIN])
    assert drift.shift(predicted, (1400.0, 300.0), error_px=100.0) is None
    assert drift.shift(predicted[:2], (540.0, 320.0), error_px=100.0) is None


def test_drift_shift_on_an_element_aims_at_its_nearest_point():
    predicted = np.tile([500.0, 300.0], (10, 1))
    inside = Rect(450, 280, 100, 40)
    np.testing.assert_allclose(drift.shift(predicted, inside, error_px=100.0), [0, 0])
    beside = Rect(560, 290, 40, 20)
    np.testing.assert_allclose(
        drift.shift(predicted, beside, error_px=100.0), [60 * drift.RECT_GAIN, 0]
    )


def test_camera_corrects_drift_from_recent_features(tmp_path):
    rng = np.random.default_rng(6)
    grid = [(x, y) for x in (100, 720, 1340) for y in (80, 450, 820)]
    fitted = model.fit(*synthetic(grid, rng))
    camera = CameraGaze(GazeHistory(), tmp_path / "face.task", tmp_path / "cal.npz")
    camera.set_model(fitted)
    probe, _, _ = synthetic([(600, 300)], rng, per_target=12)
    for i, row in enumerate(probe):
        camera._recent.append((10.0 + i / 30, row))
    predicted = np.median(fitted.predict(probe), axis=0)
    step = camera.correct_drift(9.0, 11.0, (predicted[0] + 50, predicted[1]))
    assert step is not None and step[0] > 0
    reloaded = model.GazeModel.load(tmp_path / "cal.npz")
    np.testing.assert_allclose(reloaded.offset, camera.model.offset)
    assert camera.correct_drift(20.0, 21.0, (0.0, 0.0)) is None


def test_pursuit_moves_at_a_steady_speed_along_its_path():
    size = (1000.0, 500.0)
    path = Pursuit(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)))
    assert path.seconds(size) == pytest.approx(1500 / PURSUIT_PX_S)
    assert path.at(0.0, size) == (0.0, 0.0)
    assert path.at(500 / PURSUIT_PX_S, size) == pytest.approx((500.0, 0.0))
    assert path.at(1250 / PURSUIT_PX_S, size) == pytest.approx((1000.0, 250.0))
    assert path.at(99.0, size) == (1000.0, 500.0)


def test_check_pass_picks_the_more_accurate_model():
    rng = np.random.default_rng(7)
    grid = [(x, y) for x in (100, 720, 1340) for y in (80, 450, 820)]
    good = model.fit(*synthetic(grid, rng))
    bad = replace(good, offset=np.array([200.0, 0.0]))
    checks = []
    for visit, target in enumerate([(300, 200), (1100, 200), (700, 700)]):
        rows, _, _ = synthetic([target], rng, per_target=10)
        checks += [(r, target, visit) for r in rows]
    picked, error = _pick([bad, good], checks)
    assert picked is good and error < 80
    assert check_error(good, checks[:15]) is None  # too few dots seen
    picked, error = _pick([bad, good], [])
    assert picked is bad and error is None
