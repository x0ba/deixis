"""What the user is looking at, as a snapshot the agent can reason over.

Webcam gaze is off by roughly 100-200 px, so besides the element directly
under the gaze point we sample a grid around it and offer every interactive
element we hit as a candidate, nearest first.
"""

import base64
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import Quartz

from deixis.gaze.history import Fixation
from deixis.screen import ax, desktop

log = logging.getLogger(__name__)

MAX_WINDOWS = 14
CROP_W, CROP_H = 1100, 700
CROP_MAX_PX = 1100
GRID_SPACING_PX = 26.0


@dataclass
class Registry:
    """Maps the short ids shown to the agent back to live AX objects."""

    windows: dict[str, object] = field(default_factory=dict)
    window_info: dict[str, desktop.WindowInfo] = field(default_factory=dict)
    elements: dict[str, object] = field(default_factory=dict)

    def add_element(self, element) -> str:
        key = f"e{len(self.elements) + 1}"
        self.elements[key] = element
        return key


@dataclass
class Snapshot:
    data: dict
    registry: Registry
    image_png: bytes | None = None


def _ax_windows(pid: int) -> list:
    return list(ax.attr(ax.app_element(pid), "AXWindows") or [])


def _match_ax_window(info: desktop.WindowInfo, candidates: list):
    for w in candidates:
        if ax.window_number(w) == info.number:
            return w
    for w in candidates:
        r = ax.frame(w)
        if (
            r
            and abs(r.x - info.bounds.x) < 4
            and abs(r.y - info.bounds.y) < 4
            and abs(r.w - info.bounds.w) < 4
        ):
            return w
    return None


def describe(
    element,
    registry: Registry,
    gx: float | None = None,
    gy: float | None = None,
    value_limit: int = 400,
) -> dict:
    role = ax.attr(element, "AXRole")
    rect = ax.frame(element)
    out: dict = {"id": registry.add_element(element), "role": role}
    subrole = ax.attr(element, "AXSubrole")
    if subrole:
        out["subrole"] = subrole
    name = ax.label(element)
    if name:
        out["label"] = name
    value = ax.text(ax.attr(element, "AXValue"), value_limit)
    if value:
        out["value"] = value
    selected = ax.text(ax.attr(element, "AXSelectedText"), 300)
    if selected:
        out["selected_text"] = selected
    if role in ax.TEXT_ROLES or ax.settable(element, "AXValue"):
        out["editable"] = True
    if ax.attr(element, "AXFocused"):
        out["focused"] = True
    actions = [
        a
        for a in ax.action_names(element)
        if a in ("AXPress", "AXConfirm", "AXOpen", "AXShowMenu", "AXIncrement")
    ]
    if actions:
        out["actions"] = actions
    if rect:
        out["frame"] = rect.as_dict()
        if gx is not None and gy is not None:
            out["distance_px"] = round(rect.distance(gx, gy))
    return out


def _interactive(element):
    current = element
    for _ in range(4):
        if current is None:
            return None
        if ax.attr(current, "AXRole") in ax.INTERACTIVE_ROLES:
            return current
        current = ax.attr(current, "AXParent")
    return None


def _nearby(app, gx: float, gy: float, radius: float, registry: Registry, skip) -> list[dict]:
    # Fine enough to land on single-line fields (~30 px tall) anywhere in the radius.
    n = max(2, int(radius // GRID_SPACING_PX))
    offsets = [radius * i / n for i in range(-n, n + 1)]
    seen: set[tuple] = set()
    if skip is not None:
        r = ax.frame(skip)
        seen.add((ax.attr(skip, "AXRole"), r))
    found = []
    for dy in offsets:
        for dx in offsets:
            hit = ax.element_at(app, gx + dx, gy + dy)
            element = _interactive(hit) if hit is not None else None
            if element is None:
                continue
            rect = ax.frame(element)
            key = (ax.attr(element, "AXRole"), rect)
            if key in seen or rect is None or rect.w * rect.h > 900_000:
                continue
            seen.add(key)
            found.append((rect.distance(gx, gy), element))
    found.sort(key=lambda item: item[0])
    return [describe(e, registry, gx, gy, value_limit=160) for _, e in found[:10]]


def _window_summary(key: str, info: desktop.WindowInfo, window) -> dict:
    out = {"id": key, "app": info.app, "frame": info.bounds.as_dict()}
    if window is not None:
        title = ax.text(ax.attr(window, "AXTitle"), 120)
        if title:
            out["title"] = title
        if ax.attr(window, "AXFullScreen"):
            out["fullscreen"] = True
        if ax.attr(window, "AXMain"):
            out["main"] = True
    return out


def _screenshot(gx: float, gy: float) -> tuple[bytes | None, dict | None]:
    if not Quartz.CGPreflightScreenCaptureAccess():
        return None, None
    screen = None
    for s in desktop.screens():
        if s.frame.contains(gx, gy):
            screen = s
    if screen is None:
        return None, None
    f = screen.frame
    x = min(max(gx - CROP_W / 2, f.x), f.x + f.w - CROP_W)
    y = min(max(gy - CROP_H / 2, f.y), f.y + f.h - CROP_H)
    w, h = min(CROP_W, f.w), min(CROP_H, f.h)
    import cv2

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "crop.png"
        cmd = ["screencapture", "-x", "-R", f"{int(x)},{int(y)},{int(w)},{int(h)}", str(path)]
        if subprocess.run(cmd, check=False, timeout=5).returncode != 0 or not path.exists():
            return None, None
        image = cv2.imread(str(path))
    if image is None:
        return None, None
    scale = image.shape[1] / w
    if image.shape[1] > CROP_MAX_PX:
        factor = CROP_MAX_PX / image.shape[1]
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        scale *= factor
    center = (int((gx - x) * scale), int((gy - y) * scale))
    cv2.circle(image, center, int(28 * scale), (0, 0, 255), 3)
    cv2.circle(image, center, 3, (0, 0, 255), -1)
    ok, png = cv2.imencode(".png", image)
    if not ok:
        return None, None
    meta = {"origin": {"x": round(x), "y": round(y)}, "points_per_pixel": round(1 / scale, 3)}
    return png.tobytes(), meta


def build(
    gaze: Fixation | None,
    other_fixations: list[Fixation],
    gaze_error_px: float,
    want_image: bool,
) -> Snapshot:
    registry = Registry()
    data: dict = {}

    infos = desktop.windows()[:MAX_WINDOWS]
    ax_cache: dict[int, list] = {}
    window_list = []
    by_number: dict[int, str] = {}
    for i, info in enumerate(infos, start=1):
        key = f"w{i}"
        if info.pid not in ax_cache:
            app = desktop.running_app(info.pid)
            ax.enable_web_accessibility(info.pid, app.bundleIdentifier() if app else None)
            ax_cache[info.pid] = _ax_windows(info.pid)
        window = _match_ax_window(info, ax_cache[info.pid])
        registry.windows[key] = window
        registry.window_info[key] = info
        by_number[info.number] = key
        window_list.append(_window_summary(key, info, window))
    data["windows_front_to_back"] = window_list
    data["screens"] = [
        {"frame": s.frame.as_dict(), "usable": s.visible.as_dict()} for s in desktop.screens()
    ]

    front = desktop.frontmost_app()
    if front is not None:
        data["frontmost_app"] = str(front.localizedName())
    focused = ax.attr(ax.AS.AXUIElementCreateSystemWide(), "AXFocusedUIElement")
    if focused is not None:
        data["focused_element"] = describe(focused, registry, value_limit=200)

    if gaze is None:
        data["gaze"] = None
        return Snapshot(data, registry)

    gx, gy = gaze.x, gaze.y
    data["gaze"] = {
        "x": round(gx),
        "y": round(gy),
        "dwell_ms": round(gaze.duration * 1000),
        "typical_error_px": round(gaze_error_px),
    }
    hit = desktop.window_at(gx, gy)
    if hit is not None and hit.number in by_number:
        key = by_number[hit.number]
        target: dict = {"window": key, "app": hit.app}
        app = ax.app_element(hit.pid)
        element = ax.element_at(app, gx, gy)
        if element is not None and ax.attr(element, "AXRole") not in ("AXWindow", None):
            target["element"] = describe(element, registry, gx, gy)
            inside = []
            for parent in ax.ancestors(element):
                role, name = ax.attr(parent, "AXRole"), ax.label(parent)
                inside.append(f"{role} '{name}'" if name else str(role))
            target["inside"] = inside
        radius = max(60.0, min(gaze_error_px, 220.0))
        target["nearby_elements"] = _nearby(app, gx, gy, radius, registry, element)
        data["looking_at"] = target
    else:
        data["looking_at"] = {"window": None, "note": "gaze is on the desktop or menu bar"}

    earlier = []
    for f in other_fixations:
        w = desktop.window_at(f.x, f.y)
        entry: dict = {"x": round(f.x), "y": round(f.y), "dwell_ms": round(f.duration * 1000)}
        if w is not None:
            entry["window"] = by_number.get(w.number)
            entry["app"] = w.app
            el = ax.element_at(ax.app_element(w.pid), f.x, f.y)
            if el is not None:
                entry["element"] = describe(el, registry, f.x, f.y, value_limit=120)
        earlier.append(entry)
    if earlier:
        data["other_fixations_in_order"] = earlier

    image = None
    if want_image:
        try:
            image, meta = _screenshot(gx, gy)
            if meta:
                data["screenshot"] = meta
        except Exception as err:
            log.warning("screenshot failed: %s", err)
    return Snapshot(data, registry, image)


def image_b64(png: bytes) -> str:
    return base64.b64encode(png).decode()
