"""The Deixis menu bar app: wires gaze, voice, screen context and the agent."""

import logging
import queue
import re
import sys
import threading
import time

import AppKit
import objc
from PyObjCTools import AppHelper

from deixis import wake
from deixis.agent.agent import Agent
from deixis.config import Config, load_config
from deixis.gaze.history import Fixation, GazeHistory, primary
from deixis.gaze.tracker import CameraGaze, MouseGaze
from deixis.screen import ax
from deixis.screen import context as screen_context
from deixis.ui.calibration import Calibrator
from deixis.ui.overlay import Overlay
from deixis.voice.listener import Listener, Turn
from deixis.voice.speaker import Speaker

log = logging.getLogger("deixis")

FOLLOW_UP_S = 10.0
GAZE_LEAD_S = 0.5
MOUSE_ERROR_PX = 25.0
UNCALIBRATED_ERROR_PX = 400.0
KEYTERMS = ["Deixis", "full screen", "split screen", "minimize", "text box"]
SLEEP = re.compile(r"^(go to sleep|stop listening|sleep|pause)\W*$", re.I)
WAKE = re.compile(r"^(wake up|start listening|resume|i'm back)\W*$", re.I)
CALIBRATE = re.compile(r"^(re)?calibrat(e|ion)\b.*$", re.I)
CANCEL = re.compile(r"^(never ?mind|cancel|nothing|stop)\W*$", re.I)


def _on_main(fn, *args) -> None:
    AppHelper.callAfter(fn, *args)


class MenuTarget(AppKit.NSObject):
    def initWithApp_(self, app):
        self = objc.super(MenuTarget, self).init()
        if self is None:
            return None
        self.app = app
        return self

    def togglePause_(self, sender):
        self.app.set_sleeping(not self.app.sleeping)

    def calibrate_(self, sender):
        self.app.start_calibration()

    def toggleDot_(self, sender):
        self.app.toggle_dot()

    def useCamera_(self, sender):
        self.app.switch_gaze("camera")

    def useMouse_(self, sender):
        self.app.switch_gaze("mouse")

    def openPrivacy_(self, sender):
        url = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
        AppKit.NSWorkspace.sharedWorkspace().openURL_(AppKit.NSURL.URLWithString_(url))

    def quit_(self, sender):
        self.app.quit()


class DeixisApp:
    def __init__(self, config: Config):
        self.config = config
        self.history = GazeHistory()
        self.sleeping = False
        self._follow_up_until = 0.0
        self._calibrator: Calibrator | None = None
        self._commands: queue.Queue[Turn] = queue.Queue()
        self._speaking = False
        self.gaze: CameraGaze | MouseGaze | None = None
        self.overlay: Overlay | None = None

    # setup

    def run(self) -> None:
        AppKit.NSApplication.sharedApplication()
        AppKit.NSApp.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        self.overlay = Overlay()
        self.overlay.set_dot_enabled(self.config.show_gaze_dot)
        self._build_menu()

        problems = []
        if not ax.is_trusted(prompt=True):
            problems.append("Allow Accessibility for this app, then restart Deixis.")
        if not self.config.anthropic_api_key:
            problems.append("Add ANTHROPIC_API_KEY to .env.")
        if not self.config.cartesia_api_key:
            problems.append("Add CARTESIA_API_KEY to .env.")

        self.speaker = Speaker(
            self.config.cartesia_api_key, self.config.voice_id, self._on_speaking
        )
        self.agent = Agent(self.config.anthropic_api_key, self.config.anthropic_model)
        self._start_gaze(self.config.gaze_source)
        threading.Thread(target=self._worker, name="commands", daemon=True).start()

        if self.config.cartesia_api_key:
            self.listener = Listener(
                self.config.cartesia_api_key,
                KEYTERMS,
                on_turn_start=lambda t: None,
                on_partial=self._on_partial,
                on_turn=self._on_turn,
                on_status=lambda s: _on_main(self._status_listening, s),
            )
            self.listener.start()
        else:
            self.listener = None

        if problems:
            self._show("error", problems[0])
            self.speaker.say(" ".join(problems))
        else:
            self._show("idle", "Say “Deixis” and what you'd like.")
            self._hide_later(4.0)
        if isinstance(self.gaze, CameraGaze) and not self.gaze.calibrated:
            AppHelper.callLater(2.0, self.start_calibration)
        AppHelper.runEventLoop(installInterrupt=True)

    def _build_menu(self) -> None:
        self._target = MenuTarget.alloc().initWithApp_(self)
        bar = AppKit.NSStatusBar.systemStatusBar()
        self._item = bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
        image = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_("eye", "Deixis")
        if image is not None:
            image.setTemplate_(True)
            self._item.button().setImage_(image)
        else:
            self._item.button().setTitle_("Deixis")
        menu = AppKit.NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)
        self._state_item = menu.addItemWithTitle_action_keyEquivalent_("Starting…", None, "")
        self._state_item.setEnabled_(False)
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        self._pause_item = self._add(menu, "Pause Listening", "togglePause:")
        self._add(menu, "Calibrate Eyes", "calibrate:")
        self._dot_item = self._add(menu, "Show Gaze Dot", "toggleDot:")
        self._dot_item.setState_(1 if self.config.show_gaze_dot else 0)
        source = menu.addItemWithTitle_action_keyEquivalent_("Gaze Source", None, "")
        sub = AppKit.NSMenu.alloc().init()
        self._camera_item = self._add(sub, "Webcam (eye tracking)", "useCamera:")
        self._mouse_item = self._add(sub, "Pointer (mouse, head mouse, eye tracker)", "useMouse:")
        source.setSubmenu_(sub)
        self._add(menu, "Open Privacy Settings…", "openPrivacy:")
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        self._add(menu, "Quit Deixis", "quit:", "q")
        self._item.setMenu_(menu)

    def _add(self, menu, title: str, action: str, key: str = ""):
        item = menu.addItemWithTitle_action_keyEquivalent_(title, action, key)
        item.setTarget_(self._target)
        return item

    # gaze

    def _start_gaze(self, source: str) -> None:
        if self.gaze is not None:
            self.gaze.stop()
        if source == "camera":
            self.gaze = CameraGaze(
                self.history,
                self.config.face_model_path,
                self.config.calibration_path,
                self.config.camera_index,
            )
            self.overlay.set_dot_enabled(self._dot_item.state() == 1)
        else:
            self.gaze = MouseGaze(self.history)
            self.overlay.set_dot_enabled(False)
        self.gaze.on_point = lambda x, y: _on_main(self.overlay.move_dot, x, y)
        self.gaze.start()
        self._camera_item.setState_(1 if source == "camera" else 0)
        self._mouse_item.setState_(1 if source == "mouse" else 0)
        self._state_item.setTitle_(f"Gaze: {'webcam' if source == 'camera' else 'pointer'}")

    def switch_gaze(self, source: str) -> None:
        self._start_gaze(source)
        if source == "camera" and not self.gaze.calibrated:
            self.start_calibration()

    def toggle_dot(self) -> None:
        on = self._dot_item.state() != 1
        self._dot_item.setState_(1 if on else 0)
        self.overlay.set_dot_enabled(on and isinstance(self.gaze, CameraGaze))

    def gaze_error_px(self) -> float:
        if isinstance(self.gaze, MouseGaze):
            return MOUSE_ERROR_PX
        if isinstance(self.gaze, CameraGaze) and self.gaze.model is not None:
            return self.gaze.model.cv_error_px
        return UNCALIBRATED_ERROR_PX

    def start_calibration(self) -> None:
        if not isinstance(self.gaze, CameraGaze):
            self.speaker.say("Calibration is only for webcam eye tracking.")
            return
        if self.gaze.error:
            self._show("error", f"Camera problem: {self.gaze.error}")
            return
        if self._calibrator is not None:
            return
        self.overlay.set_dot_enabled(False)
        self.speaker.say("Let's calibrate. Look at each dot until it moves.")
        self._calibrator = Calibrator(self.gaze, self._calibrated)
        self._calibrator.start()

    def _calibrated(self, model, message: str) -> None:
        self._calibrator = None
        if model is not None:
            self.gaze.set_model(model)
        self.overlay.set_dot_enabled(self._dot_item.state() == 1)
        self._show("idle" if model else "error", message)
        self._hide_later(5.0)
        self.speaker.say(message)

    # status

    def _show(self, state: str, message: str) -> None:
        if threading.current_thread() is threading.main_thread():
            self.overlay.status(state, message)
        else:
            _on_main(self.overlay.status, state, message)

    def _hide_later(self, seconds: float) -> None:
        stamp = time.monotonic()
        self._last_status = stamp

        def hide():
            if getattr(self, "_last_status", None) == stamp:
                self.overlay.status("idle", "")

        AppHelper.callLater(seconds, hide)

    def _status_listening(self, status: str) -> None:
        if status == "listening":
            self._pause_item.setTitle_("Pause Listening")
        else:
            self._show("error", status)

    def _on_speaking(self, speaking: bool) -> None:
        self._speaking = speaking
        if self.listener is not None:
            (self.listener.muted.set if speaking else self.listener.muted.clear)()
        if not speaking and self._follow_up_until:
            self._follow_up_until = time.monotonic() + FOLLOW_UP_S

    def set_sleeping(self, sleeping: bool) -> None:
        self.sleeping = sleeping
        self._pause_item.setTitle_("Resume Listening" if sleeping else "Pause Listening")
        if sleeping:
            self._show("idle", "Sleeping. Say “Deixis, wake up.”")
        else:
            self._show("listening", "Listening.")
        self._hide_later(3.0)

    # voice

    def _expecting_follow_up(self) -> bool:
        return time.monotonic() < self._follow_up_until

    def _on_partial(self, text: str) -> None:
        if self.sleeping:
            return
        command = wake.split_command(text, self.config.wake_words)
        if command is not None:
            self._show("listening", command or "Listening…")
        elif self._expecting_follow_up() or not self.config.require_wake_word:
            self._show("listening", text)

    def _on_turn(self, turn: Turn) -> None:
        log.info("heard: %s", turn.transcript)
        command = wake.split_command(turn.transcript, self.config.wake_words)
        if command is None:
            if self._expecting_follow_up() or not self.config.require_wake_word:
                command = turn.transcript.strip()
            else:
                return
        if self.sleeping:
            if WAKE.match(command):
                _on_main(self.set_sleeping, False)
                self.speaker.say("I'm listening.")
            return
        if not command:
            self._follow_up_until = time.monotonic() + FOLLOW_UP_S
            self._show("listening", "Yes?")
            self.speaker.say("Yes?")
            return
        self._follow_up_until = 0.0
        if SLEEP.match(command):
            _on_main(self.set_sleeping, True)
            self.speaker.say("Okay, I'll wait.")
            return
        if CANCEL.match(command):
            self.speaker.interrupt()
            self._show("idle", "")
            return
        if CALIBRATE.match(command):
            _on_main(self.start_calibration)
            return
        self._commands.put(Turn(command, turn.started, turn.ended))

    # commands

    def _snapshot(self, start: float, end: float) -> screen_context.Snapshot:
        fixations = self.history.fixations(start - GAZE_LEAD_S, end)
        main = primary(fixations)
        if main is None:
            latest = self.history.latest()
            if latest is not None and time.monotonic() - latest[0] < 2.0:
                main = Fixation(latest[1], latest[2], latest[0], latest[0])
        others = [f for f in fixations if f is not main][-4:]
        return screen_context.build(
            main, others, self.gaze_error_px(), want_image=self.config.send_screenshot
        )

    def _refresh(self) -> screen_context.Snapshot:
        now = time.monotonic()
        return self._snapshot(now - 0.8, now)

    def _worker(self) -> None:
        while True:
            turn = self._commands.get()
            said = turn.transcript
            self._show("working", said)
            try:
                snapshot = self._snapshot(turn.started, turn.ended)
                reply = self.agent.handle(
                    said,
                    snapshot,
                    refresh=self._refresh,
                    recalibrate=lambda: _on_main(self.start_calibration),
                    on_step=lambda step, said=said: self._show("working", f"{said} — {step}"),
                )
            except Exception as err:
                log.exception("command failed")
                reply = f"Sorry, something went wrong: {err}"
            if reply.rstrip().endswith("?"):
                self._follow_up_until = time.monotonic() + FOLLOW_UP_S
            self._show("speaking", reply)
            self.speaker.say(reply)
            _on_main(self._hide_later, 6.0)

    def quit(self) -> None:
        if self.gaze is not None:
            self.gaze.stop()
        if self.listener is not None:
            self.listener.stop()
        sys.stdout.flush()
        AppKit.NSApp.terminate_(None)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    DeixisApp(load_config()).run()
