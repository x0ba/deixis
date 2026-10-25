import random

from peek.screen.ax import Rect
from peek.screen.snap import Snapper, ring, search_rings
from peek.ui.calibration import FULL_PLAN, GRID, HEAD_GRID, RECENTER_PLAN, Dot

BUTTON = Rect(500, 300, 80, 24)
LINK = Rect(500, 380, 80, 24)


class FakeScreen:
    def __init__(self, **frames):
        self.frames = frames
        self.calls = []

    def lookup(self, x, y, reach):
        self.calls.append((x, y, reach))
        near = [(r.distance(x, y), n) for n, r in self.frames.items() if r.distance(x, y) <= reach]
        if not near:
            return None
        name = min(near)[1]
        return name, self.frames[name]

    def snapper(self, reach=100.0):
        return Snapper(lambda: reach, self.lookup, frame=self.frames.get)


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
    snapper = Snapper(lambda: 100.0, lambda x, y, reach: ("doc", line), frame=frames.get)
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


def test_calibration_plans():
    assert len({(d.fx, d.fy) for d in GRID}) == 25
    assert {(d.fx, d.fy) for d in HEAD_GRID} <= {(d.fx, d.fy) for d in GRID}
    assert [s for s in FULL_PLAN if isinstance(s, Dot)] == GRID + HEAD_GRID
    assert [s for s in RECENTER_PLAN if isinstance(s, Dot)] == [Dot(0.5, 0.5)]
