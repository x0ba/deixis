import random

from peek.screen.ax import Rect
from peek.screen.snap import Candidate, Snapper, likelihood, ring, search_rings
from peek.ui.calibration import (
    CHECK,
    FULL_PLAN,
    GRID,
    HEAD_GRID,
    PURSUIT,
    QUICK_PLAN,
    RECENTER_PLAN,
    Dot,
    Fit,
)

BUTTON = Rect(500, 300, 80, 24)
LINK = Rect(500, 380, 80, 24)


class FakeScreen:
    def __init__(self, **frames):
        self.frames = frames
        self.calls = []

    def lookup(self, x, y, reach):
        self.calls.append((x, y, reach))
        return [Candidate(n, r) for n, r in self.frames.items() if r.distance(x, y) <= reach]

    def snapper(self, error=40.0):
        return Snapper(lambda: error, self.lookup, frame=self.frames.get)


def hold(snapper, x, y, start, seconds=0.3, jitter=0.0):
    t, drawn = start, None
    while t < start + seconds:
        drawn = snapper.update(t, x + random.uniform(-jitter, jitter), y)
        snapper.step(t)
        t += 1 / 30
    return drawn, t


def test_rings_cover_each_grid_point_once():
    seen = set()
    for k in range(4):
        points = ring(k, 10.0)
        assert len(points) == (8 * k or 1)
        assert all(max(abs(x), abs(y)) == k * 10.0 for x, y in points)
        seen.update(points)
    assert len(seen) == 7 * 7


def test_search_rings_grow_outward_to_reach():
    radii = [r for r, _ in search_rings(300.0)]
    assert radii == sorted(set(radii))
    assert radii[0] == 0 and 250 < radii[-1] <= 300


def test_highlights_right_away_even_when_gaze_is_jittery():
    random.seed(1)
    snapper = FakeScreen(button=BUTTON).snapper()
    drawn, _ = hold(snapper, 540, 312, 0.0, seconds=0.1, jitter=60.0)
    assert drawn == BUTTON


def test_stays_on_last_element_when_nothing_else_is_near():
    snapper = FakeScreen(button=BUTTON).snapper()
    _, t = hold(snapper, 540, 312, 0.0)
    drawn, t = hold(snapper, 1200, 900, t)
    assert drawn == BUTTON
    assert snapper.element_during(t - 0.1, t) == "button"


def test_switches_only_when_another_element_is_clearly_closer():
    snapper = FakeScreen(button=BUTTON, link=LINK).snapper()
    _, t = hold(snapper, 540, 312, 0.0)
    drawn, t = hold(snapper, 540, 355, t)
    assert drawn == BUTTON
    drawn, t = hold(snapper, 540, 392, t)
    assert drawn == LINK
    assert snapper.element_during(0.0, 0.3) == "button"


def test_follows_a_moving_element_and_lets_go_when_it_disappears():
    screen = FakeScreen(button=BUTTON)
    snapper = screen.snapper()
    _, t = hold(snapper, 540, 312, 0.0)
    moved = Rect(500, 280, 80, 24)
    screen.frames["button"] = moved
    drawn, t = hold(snapper, 540, 300, t, seconds=0.6)
    assert drawn == moved
    del screen.frames["button"]
    drawn, _ = hold(snapper, 540, 300, t, seconds=0.6)
    assert drawn is None


def test_a_line_of_a_text_view_scrolls_with_the_view():
    frames = {"doc": Rect(100, 100, 600, 800)}
    line = Rect(105, 300, 590, 16)

    def lookup(x, y, reach):
        dy = frames["doc"].y - 100
        return [Candidate("doc", Rect(line.x, line.y + dy, line.w, line.h))]

    snapper = Snapper(lambda: 40.0, lookup, frame=frames.get)
    _, t = hold(snapper, 400, 308, 0.0)
    frames["doc"] = Rect(100, 60, 600, 800)
    drawn, _ = hold(snapper, 400, 268, t, seconds=0.6)
    assert drawn == Rect(105, 260, 590, 16)


def test_resting_gaze_is_not_looked_up_again():
    screen = FakeScreen()
    snapper = screen.snapper()
    _, t = hold(snapper, 300, 300, 0.0)
    _, t = hold(snapper, 305, 302, t)
    assert len(screen.calls) == 1
    hold(snapper, 700, 300, t)
    assert 2 <= len(screen.calls) <= 3


def test_pinned_highlight_stays_while_the_gaze_moves_on():
    snapper = FakeScreen(button=BUTTON, link=LINK).snapper()
    _, t = hold(snapper, 540, 312, 0.0)
    snapper.pin()
    drawn, t = hold(snapper, 540, 392, t)
    assert drawn == BUTTON
    assert snapper.pinned_element() == "button"
    assert snapper.element_during(t - 0.1, t) == "link"
    snapper.unpin()
    drawn, _ = hold(snapper, 540, 392, t, seconds=0.1)
    assert drawn == LINK
    assert snapper.pinned_element() is None


def test_pin_can_pick_what_was_highlighted_earlier():
    snapper = FakeScreen(button=BUTTON, link=LINK).snapper()
    _, t = hold(snapper, 540, 312, 0.0)
    _, t = hold(snapper, 540, 392, t)
    snapper.pin(0.0, 0.3)
    assert snapper.pinned_element() == "button"


def test_pinned_highlight_follows_its_element_and_hides_when_it_goes():
    screen = FakeScreen(button=BUTTON, link=LINK)
    snapper = screen.snapper()
    _, t = hold(snapper, 540, 312, 0.0)
    snapper.pin()
    moved = Rect(500, 280, 80, 24)
    screen.frames["button"] = moved
    drawn, t = hold(snapper, 540, 392, t, seconds=0.6)
    assert drawn == moved
    del screen.frames["button"]
    drawn, _ = hold(snapper, 540, 392, t, seconds=0.6)
    assert drawn is None
    assert snapper.pinned


def test_calibration_plans():
    assert len({(d.fx, d.fy) for d in GRID}) == 25
    assert {(d.fx, d.fy) for d in HEAD_GRID} <= {(d.fx, d.fy) for d in GRID}
    assert [s for s in FULL_PLAN if isinstance(s, Dot)] == GRID + HEAD_GRID + CHECK
    fit = FULL_PLAN.index(next(s for s in FULL_PLAN if isinstance(s, Fit)))
    assert FULL_PLAN.index(PURSUIT) < fit < FULL_PLAN.index(CHECK[0])
    assert all(d.check for d in CHECK)
    assert not {(d.fx, d.fy) for d in CHECK} & {(d.fx, d.fy) for d in GRID}
    assert [s for s in RECENTER_PLAN if isinstance(s, Dot)] == [Dot(0.5, 0.5)]
    quick = [s for s in QUICK_PLAN if isinstance(s, Dot)]
    assert len({(d.fx, d.fy) for d in quick}) == len(quick) == 6
    assert not any(isinstance(s, Fit) or getattr(s, "check", False) for s in QUICK_PLAN)


TOOLBAR = {f"b{i}": Rect(400 + 70 * i, 300, 60, 28) for i in range(6)}


def look(snapper, name, start, seconds, rng, noise=45.0, bias=(0.0, 0.0)):
    """Webcam-like gaze at a toolbar button: filtered (so correlated) noise plus a bias.
    Returns the highlighted names, one per frame."""
    cx, cy = TOOLBAR[name].center
    ex = ey = 0.0
    t, shown = start, []
    names = {r: n for n, r in TOOLBAR.items()}
    while t < start + seconds:
        ex = 0.8 * ex + 0.6 * rng.gauss(0, noise)
        ey = 0.8 * ey + 0.6 * rng.gauss(0, noise)
        snapper.update(t, cx + bias[0] + ex, cy + bias[1] + ey)
        snapper.step(t)
        drawn = snapper.update(t, cx + bias[0] + ex, cy + bias[1] + ey)
        shown.append(names.get(drawn))
        t += 1 / 30
    return shown, t


def switches(shown):
    return sum(1 for a, b in zip(shown, shown[1:], strict=False) if a != b)


def test_noisy_gaze_settles_on_the_button_without_flicker():
    rng = random.Random(3)
    snapper = FakeScreen(**TOOLBAR).snapper(error=60.0)
    shown, t = look(snapper, "b2", 0.0, 2.0, rng)
    assert shown[-1] == "b2"
    assert switches(shown[10:]) <= 1
    shown, t = look(snapper, "b4", t, 1.5, rng)
    first = shown.index("b4")
    assert first / 30 < 1.0
    assert shown[-1] == "b4"
    assert switches(shown[first:]) <= 2


def test_noisy_gaze_mostly_shows_the_right_button():
    rng = random.Random(7)
    hits = total = 0
    for target in ("b1", "b3", "b5", "b0", "b4"):
        snapper = FakeScreen(**TOOLBAR).snapper(error=60.0)
        shown, _ = look(snapper, target, 0.0, 2.0, rng)
        hits += sum(s == target for s in shown[15:])
        total += len(shown[15:])
    assert hits / total > 0.8


def test_likelihood_prefers_small_targets_under_the_gaze():
    small, big = Rect(100, 100, 40, 20), Rect(0, 0, 800, 400)
    assert likelihood(small, 120, 110, 30.0) > 10 * likelihood(big, 120, 110, 30.0)
    assert likelihood(small, 400, 300, 30.0) < likelihood(big, 400, 300, 30.0)


def test_slow_lookups_run_in_the_background():
    import threading
    import time as clock

    release = threading.Event()

    def slow(x, y, reach):
        release.wait(2.0)
        return [Candidate("button", BUTTON)]

    snapper = Snapper(lambda: 40.0, slow, frame={"button": BUTTON}.get)
    snapper.start()
    try:
        drawn, steps_before = None, 0
        deadline = clock.monotonic() + 3.0
        while drawn is None and clock.monotonic() < deadline:
            now = clock.monotonic()
            drawn = snapper.update(now, 540, 312)
            if not release.is_set() and snapper._voted_until > 0:
                steps_before += 1  # the snap thread kept stepping while the lookup waited
                if steps_before >= 3:
                    release.set()
            clock.sleep(1 / 30)
        assert steps_before >= 3
        assert drawn == BUTTON
    finally:
        snapper.stop()


def test_neighbours_still_switch_when_the_overall_error_is_large():
    """A big calibration error is mostly a steady offset; the jitter stays small, so
    looking one button over should still move the highlight one button over."""
    rng = random.Random(11)
    snapper = FakeScreen(**TOOLBAR).snapper(error=190.0)
    shown, t = look(snapper, "b2", 0.0, 1.5, rng, noise=20.0, bias=(10.0, -30.0))
    assert shown[-1] == "b2"
    shown, t = look(snapper, "b3", t, 1.5, rng, noise=20.0, bias=(10.0, -30.0))
    assert shown[-1] == "b3"
    first = shown.index("b3")
    assert first / 30 < 0.7
