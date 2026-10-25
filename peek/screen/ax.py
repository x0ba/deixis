"""Thin wrappers over the macOS Accessibility (AX) API.

PyObjC returns `(error, value)` tuples for AX copy functions; these helpers
return the value or None. All coordinates are global, top-left origin.
"""

import ctypes
import ctypes.util
from dataclasses import dataclass

import ApplicationServices as AS
import Quartz

AX_OK = AS.kAXErrorSuccess
TEXT_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}
INTERACTIVE_ROLES = TEXT_ROLES | {
    "AXButton",
    "AXCheckBox",
    "AXRadioButton",
    "AXPopUpButton",
    "AXMenuButton",
    "AXLink",
    "AXTab",
    "AXSlider",
    "AXCell",
    "AXRow",
    "AXMenuItem",
    "AXDisclosureTriangle",
    "AXImage",
    "AXStaticText",
    "AXHeading",
}
PASSIVE_ROLES = {"AXStaticText", "AXHeading", "AXImage"}
CHROMIUM_BUNDLES = {
    "com.google.Chrome",
    "com.brave.Browser",
    "com.microsoft.edgemac",
    "company.thebrowser.Browser",
    "com.vivaldi.Vivaldi",
}


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def distance(self, px: float, py: float) -> float:
        dx = max(self.x - px, 0.0, px - (self.x + self.w))
        dy = max(self.y - py, 0.0, py - (self.y + self.h))
        return (dx * dx + dy * dy) ** 0.5

    def as_dict(self) -> dict[str, int]:
        return {"x": round(self.x), "y": round(self.y), "w": round(self.w), "h": round(self.h)}


def is_trusted(prompt: bool = False) -> bool:
    return bool(AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: prompt}))


def attr(element, name: str):
    err, value = AS.AXUIElementCopyAttributeValue(element, name, None)
    return value if err == AX_OK else None


def attr_names(element) -> list[str]:
    err, names = AS.AXUIElementCopyAttributeNames(element, None)
    return list(names) if err == AX_OK and names else []


def action_names(element) -> list[str]:
    err, names = AS.AXUIElementCopyActionNames(element, None)
    return list(names) if err == AX_OK and names else []


def settable(element, name: str) -> bool:
    err, ok = AS.AXUIElementIsAttributeSettable(element, name, None)
    return err == AX_OK and bool(ok)


def set_attr(element, name: str, value) -> bool:
    return AS.AXUIElementSetAttributeValue(element, name, value) == AX_OK


def perform(element, action: str) -> bool:
    return AS.AXUIElementPerformAction(element, action) == AX_OK


def pid_of(element) -> int | None:
    err, pid = AS.AXUIElementGetPid(element, None)
    return pid if err == AX_OK else None


def app_element(pid: int):
    app = AS.AXUIElementCreateApplication(pid)
    AS.AXUIElementSetMessagingTimeout(app, 1.0)
    return app


def element_at(app, x: float, y: float):
    err, element = AS.AXUIElementCopyElementAtPosition(app, x, y, None)
    return element if err == AX_OK else None


def frame(element) -> Rect | None:
    pos, size = attr(element, "AXPosition"), attr(element, "AXSize")
    if pos is None or size is None:
        return None
    ok_p, point = AS.AXValueGetValue(pos, AS.kAXValueCGPointType, None)
    ok_s, dims = AS.AXValueGetValue(size, AS.kAXValueCGSizeType, None)
    if not (ok_p and ok_s):
        return None
    return Rect(point.x, point.y, dims.width, dims.height)


def _param(element, name: str, value):
    err, out = AS.AXUIElementCopyParameterizedAttributeValue(element, name, value, None)
    return out if err == AX_OK else None


def line_bounds(element, x: float, y: float) -> Rect | None:
    """Frame of the line of text at a point, for text views that support it."""
    point = AS.AXValueCreate(AS.kAXValueCGPointType, Quartz.CGPointMake(x, y))
    at = _param(element, "AXRangeForPosition", point)
    if at is None:
        return None
    ok, (index, _) = AS.AXValueGetValue(at, AS.kAXValueCFRangeType, None)
    line = _param(element, "AXLineForIndex", index) if ok else None
    span = _param(element, "AXRangeForLine", line) if line is not None else None
    bounds = _param(element, "AXBoundsForRange", span) if span is not None else None
    if bounds is None:
        return None
    ok, r = AS.AXValueGetValue(bounds, AS.kAXValueCGRectType, None)
    if not ok or r.size.width <= 0 or r.size.height <= 0:
        return None
    return Rect(r.origin.x, r.origin.y, r.size.width, r.size.height)


def set_frame(element, rect: Rect) -> bool:
    # Size first, then position, then size again: some apps clamp size to the
    # space left at the old position.
    size = AS.AXValueCreate(AS.kAXValueCGSizeType, Quartz.CGSizeMake(rect.w, rect.h))
    pos = AS.AXValueCreate(AS.kAXValueCGPointType, Quartz.CGPointMake(rect.x, rect.y))
    set_attr(element, "AXSize", size)
    moved = set_attr(element, "AXPosition", pos)
    resized = set_attr(element, "AXSize", size)
    return moved or resized


def role_and_frame(element) -> tuple[str | None, Rect | None]:
    """Role and frame in one round trip to the app."""
    err, values = AS.AXUIElementCopyMultipleAttributeValues(
        element, ["AXRole", "AXPosition", "AXSize"], 0, None
    )
    if err != AX_OK or not values or len(values) != 3:
        return None, None
    role, pos, size = values
    role = role if isinstance(role, str) else None
    try:
        ok_p, point = AS.AXValueGetValue(pos, AS.kAXValueCGPointType, None)
        ok_s, dims = AS.AXValueGetValue(size, AS.kAXValueCGSizeType, None)
    except (TypeError, ValueError):
        return role, None
    if not (ok_p and ok_s):
        return role, None
    return role, Rect(point.x, point.y, dims.width, dims.height)


def web_area(element):
    """The web page an element belongs to, if it's web content."""
    current = element
    for _ in range(60):
        if current is None:
            return None
        role = attr(current, "AXRole")
        if role == "AXWebArea":
            return current
        if role in ("AXWindow", "AXApplication"):
            return None
        current = attr(current, "AXParent")
    return None


WEB_SEARCH_KEYS = [
    "AXControlSearchKey",
    "AXLinkSearchKey",
    "AXGraphicSearchKey",
    "AXHeadingSearchKey",
    "AXStaticTextSearchKey",
]


def visible_web_elements(page, limit: int = 1500) -> list:
    """Every visible control, link, image, heading and text on a web page.

    This is the search VoiceOver's rotor uses. Chromium and WebKit answer it from
    their own tree in one call, unlike hit-testing, which Chromium answers from a
    cache of the last hit test and fixes up later.
    """
    query = {
        "AXSearchKey": WEB_SEARCH_KEYS,
        "AXVisibleOnly": True,
        "AXResultsLimit": limit,
        "AXDirection": "AXDirectionNext",
    }
    return list(_param(page, "AXUIElementsForSearchPredicate", query) or [])


def window_of(element):
    window = attr(element, "AXWindow")
    if window is not None:
        return window
    current = element
    for _ in range(40):
        if current is None:
            return None
        if attr(current, "AXRole") == "AXWindow":
            return current
        current = attr(current, "AXParent")
    return None


def interactive(element):
    """The element itself or its nearest interactive ancestor, a few levels up.

    Text or an image inside a control (a button's label, a link's icon) gives
    the control.
    """
    found, current = None, element
    for _ in range(4):
        if current is None:
            break
        role = attr(current, "AXRole")
        if role in INTERACTIVE_ROLES:
            if role not in PASSIVE_ROLES:
                return current
            found = found or current
        current = attr(current, "AXParent")
    return found


def _is_item(element, role) -> bool:
    if role == "AXRow":
        return True
    return role == "AXGroup" and attr(element, "AXSubrole") == "AXDocumentArticle"


def item_of(element, limit: int = 30):
    """The post, message or row an element is part of: the element itself or its
    nearest ancestor that is a web article or a table/list row."""
    current = element
    for _ in range(limit):
        if current is None:
            return None
        role = attr(current, "AXRole")
        if role in ("AXWindow", "AXApplication", "AXWebArea"):
            return None
        if _is_item(current, role):
            return current
        current = attr(current, "AXParent")
    return None


def controls_in(item, limit: int = 30, max_nodes: int = 400) -> list:
    """The buttons, links and fields inside `item`, in reading order. Items nested
    in it (a quoted post) are skipped: their controls act on them, not on `item`."""
    found, visited = [], 0
    stack = list(reversed(attr(item, "AXChildren") or []))
    while stack and len(found) < limit and visited < max_nodes:
        element = stack.pop()
        visited += 1
        role = attr(element, "AXRole")
        if _is_item(element, role):
            continue
        if role in INTERACTIVE_ROLES and role not in PASSIVE_ROLES and role != "AXCell":
            found.append(element)
            continue
        stack.extend(reversed(attr(element, "AXChildren") or []))
    return found


def ancestors(element, limit: int = 8) -> list:
    chain, current = [], attr(element, "AXParent")
    while current is not None and len(chain) < limit:
        if attr(current, "AXRole") in ("AXWindow", "AXApplication"):
            break
        chain.append(current)
        current = attr(current, "AXParent")
    return chain


def text(value, limit: int) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return s if len(s) <= limit else s[:limit] + "…"


def label(element) -> str | None:
    for name in ("AXTitle", "AXDescription", "AXPlaceholderValue", "AXHelp"):
        value = text(attr(element, name), 120)
        if value:
            return value
    title_el = attr(element, "AXTitleUIElement")
    if title_el is not None:
        return text(attr(title_el, "AXValue"), 120)
    return None


_SPI = None


def window_number(window) -> int | None:
    """CGWindowID of an AX window via the private `_AXUIElementGetWindow`."""
    global _SPI
    if _SPI is None:
        lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("ApplicationServices"))
        _SPI = lib._AXUIElementGetWindow
        _SPI.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        _SPI.restype = ctypes.c_int32
    import objc

    out = ctypes.c_uint32(0)
    if _SPI(objc.pyobjc_id(window), ctypes.byref(out)) != 0:
        return None
    return int(out.value) or None


_ENABLED: set[int] = set()


def enable_web_accessibility(pid: int, bundle_id: str | None) -> None:
    """Chromium and Electron hide their AX tree until asked for it."""
    if pid in _ENABLED:
        return
    _ENABLED.add(pid)
    app = app_element(pid)
    set_attr(app, "AXManualAccessibility", True)
    if bundle_id in CHROMIUM_BUNDLES:
        set_attr(app, "AXEnhancedUserInterface", True)
