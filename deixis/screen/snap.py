"""Keeps the highlight on the control or text the user is most likely looking at.

Webcam gaze lands 50-150 px from what the user is looking at and jitters by
about as much, and on a screen that is nearly always a control or a piece of
text. So the highlight always sits on something: the element nearest the
smoothed gaze, kept until another one is clearly closer.
"""

import statistics
import threading
import time
from collections import deque
from collections.abc import Callable

from deixis.screen import ax, desktop
from deixis.screen.ax import Rect

SMOOTH_S = 0.25
REQUERY_PX = 16.0
RECHECK_S = 0.4
SWITCH_PX = 16.0
NEAR_SPACING_PX, NEAR_RINGS = 24.0, 5
FAR_SPACING_PX = 48.0
MAX_W, MAX_H = 500.0, 160.0
LARGE_PENALTY_PX = 40.0
MAX_WINDOW_SHARE = 0.5
CONTROL_BONUS_PX = 12.0
PAGE_CACHE_S = 0.3
KEEP_S = 30.0

Lookup = Callable[[float, float, float], tuple[object, Rect] | None]


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


def nearest_element(gx: float, gy: float, reach: float) -> tuple[object, Rect] | None:
    """The best element for the highlight in the window under the gaze, if any."""
    hit = desktop.window_at(gx, gy)
    if hit is None:
        return None
    running = desktop.running_app(hit.pid)
    ax.enable_web_accessibility(hit.pid, running.bundleIdentifier() if running else None)
    return nearest_in(ax.app_element(hit.pid), hit.bounds, gx, gy, reach)


def nearest_in(app, window: Rect, gx: float, gy: float, reach: float):
    """The closest interactive element to the gaze in one app window.

    Controls beat text at the same distance, and large elements (a paragraph,
    a big text area) only win when nothing smaller is close. In a large text
    view the line under the gaze stands in for the whole view, and other
    elements covering most of the window are never picked, so the highlight
    never outlines a whole window.
    """
    page = ax.web_area(ax.element_at(app, gx, gy))
    if page is not None:
        return _nearest_on_page(page, window, gx, gy, reach)
    return _nearest_by_hit_testing(app, window, gx, gy, reach)


def _score(role: str | None, rect: Rect, gx: float, gy: float) -> float:
    score = rect.distance(gx, gy)
    if role not in ax.PASSIVE_ROLES:
        score -= CONTROL_BONUS_PX
    if rect.w > MAX_W or rect.h > MAX_H:
        score += LARGE_PENALTY_PX
    return score


def _nearest_on_page(page, window: Rect, gx: float, gy: float, reach: float):
    max_area = MAX_WINDOW_SHARE * window.w * window.h
    best: tuple[float, object, Rect] | None = None
    for element, role, rect in _page_elements(page):
        if rect.w * rect.h > max_area or rect.distance(gx, gy) > reach:
            continue
        if not window.contains(*rect.center):
            continue
        score = _score(role, rect, gx, gy)
        if best is None or score < best[0]:
            best = (score, element, rect)
    return (best[1], best[2]) if best else None


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


def _nearest_by_hit_testing(app, window: Rect, gx: float, gy: float, reach: float):
    best: tuple[float, object, Rect] | None = None
    seen: set[tuple] = set()
    max_area = MAX_WINDOW_SHARE * window.w * window.h
    for radius, offsets in search_rings(reach):
        for dx, dy in offsets:
            px, py = gx + dx, gy + dy
            if not window.contains(px, py):
                continue
            element = ax.interactive(ax.element_at(app, px, py))
            if element is None:
                continue
            role, rect = ax.attr(element, "AXRole"), ax.frame(element)
            if rect is None or (role, rect) in seen:
                continue
            seen.add((role, rect))
            if rect.w > MAX_W or rect.h > MAX_H:
                rect = ax.line_bounds(element, px, py) or rect
            if rect.w * rect.h > max_area:
                continue
            score = _score(role, rect, gx, gy)
            if best is None or score < best[0]:
                best = (score, element, rect)
        if best is not None and best[0] <= radius:
            break
    return (best[1], best[2]) if best else None


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
    """Fed every gaze point; says what to highlight and remembers what was highlighted."""

    def __init__(
        self,
        reach: Callable[[], float],
        lookup: Lookup = nearest_element,
        frame: Callable[[object], Rect | None] = ax.frame,
    ):
        self._reach = reach
        self._lookup = lookup
        self._frame = frame
        self._points: deque[tuple[float, float, float]] = deque()
        self._current: _Snap | None = None
        self._past: deque[_Snap] = deque()
        self._queried_at: tuple[float, float] | None = None
        self._checked_at = 0.0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="gaze-snap", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def update(self, t: float, x: float, y: float) -> Rect | None:
        """Record a gaze point and return the frame to highlight, if any."""
        with self._lock:
            self._points.append((t, x, y))
            while self._points[0][0] < t - 1.0:
                self._points.popleft()
            current = self._current
        self._wake.set()
        return current.rect if current else None

    def element_during(self, start: float, end: float):
        """The element highlighted longest between `start` and `end`, if any."""
        with self._lock:
            snaps = [*self._past, *([self._current] if self._current else [])]
        best = max(snaps, key=lambda s: s.overlap(start, end), default=None)
        return best.element if best is not None and best.overlap(start, end) > 0 else None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.5)
            self._wake.clear()
            self.step(time.monotonic())

    def step(self, now: float) -> None:
        with self._lock:
            points = [p for p in self._points if p[0] >= now - SMOOTH_S]
            current = self._current
        if not points:
            return
        cx = statistics.median(p[1] for p in points)
        cy = statistics.median(p[2] for p in points)
        if current is not None:
            current.end = now
            if now - self._checked_at >= RECHECK_S:
                current = self._recheck(current, now)
            if current is not None and current.rect.contains(cx, cy):
                return
        if self._queried_at is not None:
            qx, qy = self._queried_at
            if ((qx - cx) ** 2 + (qy - cy) ** 2) ** 0.5 <= REQUERY_PX:
                return
        self._queried_at = (cx, cy)
        found = self._lookup(cx, cy, self._reach())
        if found is None:
            return
        element, rect = found
        if current is not None:
            if rect == current.rect:
                return
            if rect.distance(cx, cy) + SWITCH_PX > current.rect.distance(cx, cy):
                return
        self._replace(_Snap(element, rect, self._frame(element), now), now)

    def _recheck(self, current: _Snap, now: float) -> _Snap | None:
        """Follows the highlighted element when it moves and lets go when it's gone."""
        self._checked_at = now
        self._queried_at = None
        frame = self._frame(current.element)
        if frame is None or frame.w <= 0 or frame.h <= 0:
            self._replace(None, now)
            return None
        if frame != current.frame:
            r, old = current.rect, current.frame
            moved = Rect(r.x + frame.x - old.x, r.y + frame.y - old.y, r.w, r.h)
            with self._lock:
                current.rect = frame if r == old else moved
                current.frame = frame
        return current

    def _replace(self, snap: _Snap | None, now: float) -> None:
        with self._lock:
            if self._current is not None:
                self._current.end = now
                self._past.append(self._current)
            self._current = snap
            while self._past and self._past[0].end < now - KEEP_S:
                self._past.popleft()
