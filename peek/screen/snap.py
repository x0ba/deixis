"""Keeps the highlight on the control or text the user is most likely looking at.

Webcam gaze lands 50-200 px from what the user is looking at, but most of
that is a slowly changing offset; from moment to moment it only jitters by
20-40 px. So moving the eyes one button over moves the gaze about one button
over, even when it's offset, and the user can steer the highlight by eye.
Choosing "the element nearest the gaze" makes it flicker with the jitter.
Instead every gaze sample votes for the elements around it, each in proportion
to how likely it is that the user is looking at that element given how much
the gaze is jittering right now (small, close targets get most of the vote; a
big element only gets a share in proportion to how much of it the gaze could
be on). Votes are spread by the measured jitter, not the overall error:
spreading them by the overall error splits them evenly between neighbours,
and the highlight then gets stuck. Votes fade over a fraction of a second,
and the highlight only moves when another element has clearly out-voted the
current one. A sudden jump in gaze (a saccade) clears old votes faster, so
looking somewhere new is still picked up quickly.

While Peek handles a request the highlight can be pinned: it stays on the
element the user was looking at when they spoke, so they can look elsewhere
while Peek works on it. Gaze is still tracked underneath.
"""

import math
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from peek.screen import ax, desktop
from peek.screen.ax import Rect

REACH_PX = (150.0, 360.0)  # how far from the gaze to look for elements
MIN_SIGMA_PX = 15.0
RECHECK_S = 0.4
CANDIDATES_TTL_S = 1.5
NEAR_SPACING_PX, NEAR_RINGS = 24.0, 5
FAR_SPACING_PX = 48.0
MAX_W, MAX_H = 500.0, 160.0
MAX_WINDOW_SHARE = 0.5
MAX_CANDIDATES = 40
PAGE_CACHE_S = 0.3
KEEP_S = 30.0

CONTROL_PRIOR = 1.5  # a control is this much likelier to be the target than text
JITTER_START_PX = 35.0  # per-axis jitter assumed until it's been measured
JITTER_WINDOW_S = 0.3
JITTER_TAU_S = 2.0  # how quickly the jitter estimate follows changes
JITTER_SCALE = 1.5  # votes spread this much wider than the measured jitter
VOTE_TAU_S = 0.5  # votes fade with this time constant
MIN_VOTES = 1.5  # about 50 ms of undivided gaze before anything is highlighted
SWITCH_RATIO = 1.5  # a new element needs this many times the current one's votes
JUMP_SIGMAS = 2.5  # a gaze move this big is a new look, not jitter
JUMP_KEEP = 0.2  # share of old votes kept after a jump
JUMP_RECENT_S, JUMP_BEFORE_S = 0.1, 0.3


@dataclass(frozen=True)
class Candidate:
    element: object
    rect: Rect
    control: bool = True


Lookup = Callable[[float, float, float], list[Candidate]]


def ring(k: int, spacing: float) -> list[tuple[float, float]]:
    """Grid offsets on the square ring `k` steps out from the center."""
    if k == 0:
        return [(0.0, 0.0)]
    r = k * spacing
    side = [spacing * i for i in range(-k, k + 1)]
    inner = side[1:-1]
    return (
        [(x, -r) for x in side]
        + [(x, r) for x in side]
        + [(-r, y) for y in inner]
        + [(r, y) for y in inner]
    )


def search_rings(reach: float) -> list[tuple[float, list[tuple[float, float]]]]:
    """(ring radius, offsets) out to `reach`: dense near the gaze, sparser beyond."""
    near = min(NEAR_RINGS, int(reach // NEAR_SPACING_PX))
    rings = [(k * NEAR_SPACING_PX, ring(k, NEAR_SPACING_PX)) for k in range(near + 1)]
    k = int(near * NEAR_SPACING_PX // FAR_SPACING_PX) + 1
    while k * FAR_SPACING_PX <= reach:
        rings.append((k * FAR_SPACING_PX, ring(k, FAR_SPACING_PX)))
        k += 1
    return rings


def reach_for(error_px: float) -> float:
    low, high = REACH_PX
    return min(max(3 * error_px, low), high)


def sigma_for(error_px: float) -> float:
    """Per-axis gaze spread from the average distance error (a 2D Gaussian's mean
    radius is sigma * sqrt(pi / 2)). The most the jitter can be."""
    return max(error_px / math.sqrt(math.pi / 2), MIN_SIGMA_PX)


def spread(points: list[tuple[float, float, float]]) -> float | None:
    """Robust per-axis spread of gaze points around their median, or None if too few."""
    if len(points) < 5:
        return None
    sds = []
    for axis in (1, 2):
        values = [p[axis] for p in points]
        mid = statistics.median(values)
        sds.append(1.4826 * statistics.median(abs(v - mid) for v in values))
    return sum(sds) / 2


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _axis(lo: float, size: float, g: float, sigma: float) -> float:
    if size <= 1e-6:
        z = (g - lo) / sigma
        return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))
    return max(_phi((lo + size - g) / sigma) - _phi((lo - g) / sigma), 0.0) / size


def likelihood(rect: Rect, gx: float, gy: float, sigma: float) -> float:
    """Density of a gaze sample at (gx, gy) if the user looks somewhere in `rect`:
    a uniform spot in the rect, blurred by Gaussian gaze noise."""
    return _axis(rect.x, rect.w, gx, sigma) * _axis(rect.y, rect.h, gy, sigma)


def background(sigma: float, reach: float) -> float:
    """The density of a point target `reach` away: elements farther than that from
    the gaze get no say, and anything nearer is in the running."""
    z = min(reach / sigma, 30.0)
    return math.exp(-0.5 * z * z) / (2 * math.pi * sigma * sigma)


def candidates_near(gx: float, gy: float, reach: float) -> list[Candidate]:
    """Elements in the window under the gaze that the highlight could go on."""
    hit = desktop.window_at(gx, gy)
    if hit is None:
        return []
    running = desktop.running_app(hit.pid)
    ax.enable_web_accessibility(hit.pid, running.bundleIdentifier() if running else None)
    return candidates_in(ax.app_element(hit.pid), hit.bounds, gx, gy, reach)


def candidates_in(app, window: Rect, gx: float, gy: float, reach: float) -> list[Candidate]:
    """Interactive elements and text near the gaze in one app window.

    In a large text view the line under each probe stands in for the whole
    view, and elements covering most of the window are skipped, so the
    highlight never outlines a whole window.
    """
    page = ax.web_area(ax.element_at(app, gx, gy))
    if page is not None:
        found = _on_page(page, window, gx, gy, reach)
    else:
        found = _by_hit_testing(app, window, gx, gy, reach)
    found.sort(key=lambda c: c.rect.distance(gx, gy))
    return found[:MAX_CANDIDATES]


def _on_page(page, window: Rect, gx: float, gy: float, reach: float) -> list[Candidate]:
    max_area = MAX_WINDOW_SHARE * window.w * window.h
    found = []
    for element, role, rect in _page_elements(page):
        if rect.w * rect.h > max_area or rect.distance(gx, gy) > reach:
            continue
        if not window.contains(*rect.center):
            continue
        found.append(Candidate(element, rect, role not in ax.PASSIVE_ROLES))
    return found


_page_cache: tuple[object, float, list] | None = None


def _page_elements(page) -> list[tuple[object, str | None, Rect]]:
    """The page's visible elements with frames, reused for PAGE_CACHE_S."""
    global _page_cache
    now = time.monotonic()
    if _page_cache is not None and _page_cache[0] == page and now - _page_cache[1] < PAGE_CACHE_S:
        return _page_cache[2]
    items = []
    for element in ax.visible_web_elements(page):
        role, rect = ax.role_and_frame(element)
        if rect is not None and rect.w > 0 and rect.h > 0:
            items.append((element, role, rect))
    _page_cache = (page, now, items)
    return items


def _by_hit_testing(app, window: Rect, gx: float, gy: float, reach: float) -> list[Candidate]:
    """Probes rings around the gaze, stopping a third of `reach` past the closest find."""
    found: list[Candidate] = []
    seen: set[tuple] = set()
    closest = math.inf
    max_area = MAX_WINDOW_SHARE * window.w * window.h
    for radius, offsets in search_rings(reach):
        if radius > closest + reach / 3:
            break
        for dx, dy in offsets:
            px, py = gx + dx, gy + dy
            if not window.contains(px, py):
                continue
            element = ax.interactive(ax.element_at(app, px, py))
            if element is None:
                continue
            role, rect = ax.attr(element, "AXRole"), ax.frame(element)
            if rect is None:
                continue
            if rect.w > MAX_W or rect.h > MAX_H:
                rect = ax.line_bounds(element, px, py) or rect
            if (role, rect) in seen or rect.w * rect.h > max_area:
                continue
            seen.add((role, rect))
            found.append(Candidate(element, rect, role not in ax.PASSIVE_ROLES))
            closest = min(closest, rect.distance(gx, gy))
    return found


class _Snap:
    """`rect` is what's outlined: the element's `frame`, or one line of it."""

    def __init__(self, element, rect: Rect, frame: Rect | None, start: float):
        self.element = element
        self.rect = rect
        self.frame = frame or rect
        self.start = start
        self.end = start

    def overlap(self, start: float, end: float) -> float:
        return max(0.0, min(self.end, end) - max(self.start, start))


class Snapper:
    """Fed every gaze point; says what to highlight and remembers what was highlighted.

    `error_px` is the gaze's typical distance error, which sets how far votes spread.
    """

    def __init__(
        self,
        error_px: Callable[[], float],
        lookup: Lookup = candidates_near,
        frame: Callable[[object], Rect | None] = ax.frame,
    ):
        self._error_px = error_px
        self._lookup = lookup
        self._frame = frame
        self._points: deque[tuple[float, float, float]] = deque()
        self._current: _Snap | None = None
        self._pinned: _Snap | None = None
        self._pinning = False
        self._past: deque[_Snap] = deque()
        self._candidates: list[Candidate] = []
        self._queried: tuple[float, float, float] | None = None  # x, y, time
        self._votes: dict[Rect, float] = {}
        self._voted_until = -math.inf
        self._jump_at = -math.inf
        self._jitter = JITTER_START_PX
        self._checked_at = 0.0
        self._async = False
        self._request: tuple[float, float, float] | None = None  # x, y, reach
        self._result: list[Candidate] | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._lookup_wake = threading.Event()
        self._stop = threading.Event()

    def start(self) -> None:
        """Run on background threads. Accessibility lookups can take a few hundred ms
        in some apps, so they get their own thread and votes keep counting meanwhile."""
        self._async = True
        threading.Thread(target=self._run, name="gaze-snap", daemon=True).start()
        threading.Thread(target=self._lookups, name="gaze-lookup", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._lookup_wake.set()

    def update(self, t: float, x: float, y: float) -> Rect | None:
        """Record a gaze point and return the frame to highlight, if any."""
        with self._lock:
            self._points.append((t, x, y))
            while self._points[0][0] < t - 1.0:
                self._points.popleft()
            shown = self._pinned if self._pinning else self._current
        self._wake.set()
        return shown.rect if shown else None

    @property
    def pinned(self) -> bool:
        return self._pinning

    def pinned_element(self):
        """The pinned element, or None if nothing is pinned (or it went away)."""
        with self._lock:
            return self._pinned.element if self._pinning and self._pinned else None

    def pin(self, start: float | None = None, end: float | None = None) -> None:
        """Hold the highlight on what's highlighted now, or on the element
        highlighted longest between `start` and `end`."""
        with self._lock:
            snap = self._current
            if start is not None and end is not None:
                snap = self._longest(start, end) or snap
            self._pinned = _Snap(snap.element, snap.rect, snap.frame, snap.start) if snap else None
            self._pinning = True

    def unpin(self) -> None:
        with self._lock:
            self._pinned = None
            self._pinning = False

    def element_during(self, start: float, end: float):
        """The element highlighted longest between `start` and `end`, if any."""
        with self._lock:
            best = self._longest(start, end)
        return best.element if best else None

    def rect_during(self, start: float, end: float) -> Rect | None:
        """The outline shown longest between `start` and `end`, if any."""
        with self._lock:
            best = self._longest(start, end)
        return best.rect if best else None

    def _longest(self, start: float, end: float) -> _Snap | None:
        snaps = [*self._past, *([self._current] if self._current else [])]
        best = max(snaps, key=lambda s: s.overlap(start, end), default=None)
        return best if best is not None and best.overlap(start, end) > 0 else None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.5)
            self._wake.clear()
            self.step(time.monotonic())

    def step(self, now: float) -> None:
        with self._lock:
            points = list(self._points)
        fresh = [p for p in points if p[0] > self._voted_until]
        if not fresh:
            return
        if now - self._checked_at >= RECHECK_S:
            self._checked_at = now
            self._recheck_current(now)
            self._recheck_pinned()
        current = self._current
        if current is not None:
            current.end = now

        error = self._error_px()
        reach = reach_for(error)
        self._measure_jitter(fresh, points)
        sigma = min(max(JITTER_SCALE * self._jitter, MIN_SIGMA_PX), sigma_for(error))
        cx = statistics.median(p[1] for p in fresh)
        cy = statistics.median(p[2] for p in fresh)
        self._refresh_candidates(cx, cy, now, sigma, reach)
        self._decay(fresh, points, sigma)
        self._vote(fresh, sigma, reach)
        self._voted_until = fresh[-1][0]
        self._choose(now)

    def _measure_jitter(self, fresh, points) -> None:
        """Track how much the gaze wobbles while it rests, skipping just after a jump."""
        last = fresh[-1][0]
        if last - self._jump_at < 2 * JITTER_WINDOW_S:
            return
        measured = spread([p for p in points if p[0] > last - JITTER_WINDOW_S])
        if measured is None:
            return
        dt = last - max(self._voted_until, fresh[0][0] - 1 / 30)
        a = 1.0 - math.exp(-dt / JITTER_TAU_S)
        self._jitter += a * (measured - self._jitter)

    def _fresh_enough(self, cx, cy, now, sigma) -> bool:
        """Whether the last lookup was recent and near enough to still cover the gaze."""
        if self._queried is None:
            return False
        qx, qy, qt = self._queried
        moved = math.hypot(qx - cx, qy - cy)
        return moved <= max(24.0, sigma) and now - qt < CANDIDATES_TTL_S

    def _refresh_candidates(self, cx, cy, now, sigma, reach) -> None:
        if not self._async:
            if not self._fresh_enough(cx, cy, now, sigma):
                self._queried = (cx, cy, now)
                self._candidates = list(self._lookup(cx, cy, reach))
            return
        with self._lock:
            result, self._result = self._result, None
            busy = self._request is not None
        if result is not None:
            self._candidates = result
        if not busy and not self._fresh_enough(cx, cy, now, sigma):
            self._queried = (cx, cy, now)
            with self._lock:
                self._request = (cx, cy, reach)
            self._lookup_wake.set()

    def _lookups(self) -> None:
        while not self._stop.is_set():
            self._lookup_wake.wait(0.5)
            self._lookup_wake.clear()
            with self._lock:
                request = self._request
            if request is None:
                continue
            try:
                found = list(self._lookup(*request))
            except Exception:
                found = []
            with self._lock:
                self._result = found
                self._request = None
            self._wake.set()

    def _decay(self, fresh, points, sigma) -> None:
        """Fade old votes; much faster right after the gaze jumps somewhere new."""
        dt = fresh[-1][0] - max(self._voted_until, fresh[0][0] - 1 / 30)
        keep = math.exp(-dt / VOTE_TAU_S)
        if self._jumped(points, sigma):
            keep *= JUMP_KEEP
        for key in list(self._votes):
            self._votes[key] *= keep
            if self._votes[key] < 0.01:
                del self._votes[key]

    def _jumped(self, points, sigma) -> bool:
        """Whether the last JUMP_RECENT_S of gaze sits far from the JUMP_BEFORE_S before
        it. Fires once per jump."""
        last = points[-1][0]
        if last - self._jump_at < JUMP_BEFORE_S:
            return False
        recent = [p for p in points if p[0] > last - JUMP_RECENT_S]
        before = [
            p for p in points if last - JUMP_RECENT_S - JUMP_BEFORE_S < p[0] <= last - JUMP_RECENT_S
        ]
        if len(recent) < 2 or len(before) < 3:
            return False
        dx = statistics.median(p[1] for p in recent) - statistics.median(p[1] for p in before)
        dy = statistics.median(p[2] for p in recent) - statistics.median(p[2] for p in before)
        if math.hypot(dx, dy) <= JUMP_SIGMAS * sigma:
            return False
        self._jump_at = last
        return True

    def _vote(self, fresh, sigma, reach) -> None:
        if not self._candidates:
            return
        floor = background(sigma, reach)
        for _, x, y in fresh:
            weights = [
                likelihood(c.rect, x, y, sigma) * (CONTROL_PRIOR if c.control else 1.0)
                for c in self._candidates
            ]
            total = sum(weights) + floor
            for c, w in zip(self._candidates, weights, strict=True):
                if w > 0:
                    self._votes[c.rect] = self._votes.get(c.rect, 0.0) + w / total

    def _choose(self, now: float) -> None:
        best = max(self._candidates, key=lambda c: self._votes.get(c.rect, 0.0), default=None)
        if best is None:
            return
        votes = self._votes.get(best.rect, 0.0)
        current = self._current
        if votes < MIN_VOTES:
            return
        if current is not None:
            if best.rect == current.rect:
                return
            if votes < SWITCH_RATIO * self._votes.get(current.rect, 0.0):
                return
        frame = self._frame(best.element)
        if frame is None:
            self._queried = None
            return
        self._replace(_Snap(best.element, best.rect, frame, now), now)

    def _recheck_current(self, now: float) -> None:
        current = self._current
        if current is not None and not self._follow(current):
            self._votes.pop(current.rect, None)
            self._queried = None
            self._replace(None, now)

    def _recheck_pinned(self) -> None:
        """The pin stays put when its element goes away; there's just nothing to outline."""
        pinned = self._pinned
        if pinned is not None and not self._follow(pinned):
            with self._lock:
                if self._pinned is pinned:
                    self._pinned = None

    def _follow(self, snap: _Snap) -> bool:
        """Moves `snap` with its element; False if the element is gone."""
        frame = self._frame(snap.element)
        if frame is None or frame.w <= 0 or frame.h <= 0:
            return False
        if frame != snap.frame:
            r, old = snap.rect, snap.frame
            moved = Rect(r.x + frame.x - old.x, r.y + frame.y - old.y, r.w, r.h)
            with self._lock:
                snap.rect = frame if r == old else moved
                snap.frame = frame
            if snap is self._current:
                self._votes[snap.rect] = self._votes.pop(r, 0.0)
                self._queried = None
        return True

    def _replace(self, snap: _Snap | None, now: float) -> None:
        with self._lock:
            if self._current is not None:
                self._current.end = now
                self._past.append(self._current)
            self._current = snap
            while self._past and self._past[0].end < now - KEEP_S:
                self._past.popleft()
