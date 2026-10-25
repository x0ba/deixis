"""The Peek menu bar app: wires gaze, voice, screen context and the agent."""

import logging
import queue
import random
import re
import sys
import threading
import time
from dataclasses import dataclass

import AppKit
import objc
from PyObjCTools import AppHelper

from peek import wake
from peek.actions import input as inp
from peek.agent.agent import SILENT, Agent
from peek.agent.tools import Answer
from peek.config import Config, load_config
from peek.gaze.history import Fixation, GazeHistory, primary
from peek.gaze.tracker import CameraGaze, MouseGaze
from peek.screen import ax
from peek.screen import context as screen_context
from peek.screen.ax import Rect
from peek.screen.snap import Snapper
from peek.ui.calibration import Calibrator
from peek.ui.overlay import Overlay
from peek.voice.echo import content_words, is_backchannel, strip_echo
from peek.voice.listener import Listener, Turn
from peek.voice.speaker import Speaker

log = logging.getLogger("peek")

QUESTION_S = 30.0  # how long the conversation stays open after Peek asks something
ASK_TIMEOUT_S = 30.0
GAZE_LEAD_S = 0.5
MOUSE_ERROR_PX = 25.0
UNCALIBRATED_ERROR_PX = 400.0
SNAP_REACH_PX = (150.0, 360.0)
WINK_LOOKBACK_S = 0.4  # what was highlighted just before the eye shut
KEYTERMS = ["Peek", "full screen", "split screen", "minimize", "text box", "re-center"]
SLEEP = re.compile(r"^(go to sleep|stop listening|sleep|pause)\W*$", re.I)
WAKE = re.compile(r"^(wake up|start listening|resume|i'm back)\W*$", re.I)
CALIBRATE = re.compile(r"^(re)?calibrat(e|ion)\b.*$", re.I)
RECENTER = re.compile(r"^re-?cent(er|re)\b.*$", re.I)
STOP = re.compile(
    r"^((ok(ay)?|no|hey|oh)\W+)?"
    r"(stop|wait|hold on|hang on|never ?mind|cancel|nothing|quiet|be quiet|shh+|shush|"
    r"shut up|enough|that's enough)"
    r"(\W+(it|that|please|a sec(ond)?|a minute|talking))?\W*$",
    re.I,
)
DISMISS = re.compile(
    r"^((ok(ay)?|great|perfect|cool)\W+)?"
    r"(thanks|thank you|thank you so much|thanks a lot|that's all|that's it|that's everything|"
    r"i'm done|all done|we're done|bye|goodbye|see you|later)"
    r"(\W+(peek|for now|then|so much))?\W*$",
    re.I,
)
ACKNOWLEDGE = ["Anytime.", "You got it.", "Sure thing.", "Happy to help."]
STOPPED = "They told you to stop, so you stopped. Don't resume it unless they ask."
REPLACED = "They interrupted you with a new request before you finished."


def _on_main(fn, *args) -> None:
    AppHelper.callAfter(fn, *args)


def _play(name: str) -> None:
    sound = AppKit.NSSound.soundNamed_(name)
    if sound is not None:
        sound.play()


@dataclass
class Job:
    command: str
    started: float
    ended: float
    said_name: bool
    cancel_reason: str | None = None


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

    def recenter_(self, sender):
        self.app.start_calibration(quick=True)

    def toggleHighlight_(self, sender):
        self.app.toggle_highlight()

    def useCamera_(self, sender):
        self.app.switch_gaze("camera")

    def useMouse_(self, sender):
        self.app.switch_gaze("mouse")

    def openPrivacy_(self, sender):
        url = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
        AppKit.NSWorkspace.sharedWorkspace().openURL_(AppKit.NSURL.URLWithString_(url))

    def quit_(self, sender):
        self.app.quit()


class GazeApp:
    def __init__(self, config: Config):
        self.config = config
        self.history = GazeHistory()
        self.sleeping = False
        self._calibrator: Calibrator | None = None
        self._commands: queue.Queue[Job] = queue.Queue()
        self._jobs: list[Job] = []  # queued or running
        self._jobs_lock = threading.Lock()
        self._answers: queue.Queue[Turn | None] | None = None
        self._conversing = False
        self._conversation_deadline = 0.0
        self._conversation_ended = 0.0
        self._last_reply = ""
        self._name_heard = False  # in the turn being spoken
        self._held_for_request = False  # said just "Peek": keep the pin for what comes next
        self._status_stamp = 0
        self.listener: Listener | None = None
        self.gaze: CameraGaze | MouseGaze | None = None
        self.snapper: Snapper | None = None
        self._highlighted: tuple[Rect | None, bool] = (None, False)
        self.overlay: Overlay | None = None

    # setup

    def run(self) -> None:
        AppKit.NSApplication.sharedApplication()
        AppKit.NSApp.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        self.overlay = Overlay()
        self.overlay.set_highlight_enabled(self.config.show_highlight)
        self._build_menu()

        problems = []
        if not ax.is_trusted(prompt=True):
            problems.append("Allow Accessibility for this app, then restart Peek.")
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
                noise_gate_db=self.config.noise_gate_db,
            )
            self.listener.start()

        if problems:
            self._show("error", problems[0])
            self.speaker.say(" ".join(problems))
        else:
            self._show("idle", "Say “Peek” and what you'd like.")
            self._hide_later(4.0)
        if isinstance(self.gaze, CameraGaze) and not self.gaze.calibrated:
            AppHelper.callLater(2.0, self.start_calibration)
        AppHelper.callLater(0.5, self._tick)
        AppHelper.runEventLoop(installInterrupt=True)

    def _build_menu(self) -> None:
        self._target = MenuTarget.alloc().initWithApp_(self)
        bar = AppKit.NSStatusBar.systemStatusBar()
        self._item = bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
        image = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_("eye", "Peek")
        if image is not None:
            image.setTemplate_(True)
            self._item.button().setImage_(image)
        else:
            self._item.button().setTitle_("Peek")
        menu = AppKit.NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)
        self._state_item = menu.addItemWithTitle_action_keyEquivalent_("Starting…", None, "")
        self._state_item.setEnabled_(False)
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        self._pause_item = self._add(menu, "Pause Listening", "togglePause:")
        self._add(menu, "Calibrate Eyes", "calibrate:")
        self._add(menu, "Re-center Eyes", "recenter:")
        self._highlight_item = self._add(menu, "Show Highlight", "toggleHighlight:")
        self._highlight_item.setState_(1 if self.config.show_highlight else 0)
        source = menu.addItemWithTitle_action_keyEquivalent_("Gaze Source", None, "")
        sub = AppKit.NSMenu.alloc().init()
        self._camera_item = self._add(sub, "Webcam (eye tracking)", "useCamera:")
        self._mouse_item = self._add(sub, "Pointer (mouse, head mouse, eye tracker)", "useMouse:")
        source.setSubmenu_(sub)
        self._add(menu, "Open Privacy Settings…", "openPrivacy:")
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        self._add(menu, "Quit Peek", "quit:", "q")
        self._item.setMenu_(menu)

    def _add(self, menu, title: str, action: str, key: str = ""):
        item = menu.addItemWithTitle_action_keyEquivalent_(title, action, key)
        item.setTarget_(self._target)
        return item

    # gaze

    def _start_gaze(self, source: str) -> None:
        if self.gaze is not None:
            self.gaze.stop()
        if self.snapper is not None:
            self.snapper.stop()
        self._highlighted = (None, False)
        self.overlay.highlight(None)
        if source == "camera":
            self.gaze = CameraGaze(
                self.history,
                self.config.face_model_path,
                self.config.calibration_path,
                self.config.camera_index,
            )
        else:
            self.gaze = MouseGaze(self.history)
        low, high = SNAP_REACH_PX
        self.snapper = Snapper(lambda: min(max(3 * self.gaze_error_px(), low), high))
        self.snapper.start()
        self.gaze.on_point = self._on_gaze_point
        if isinstance(self.gaze, CameraGaze) and self.config.wink_click:
            self.gaze.on_wink = self._on_wink
        self.gaze.start()
        self._camera_item.setState_(1 if source == "camera" else 0)
        self._mouse_item.setState_(1 if source == "mouse" else 0)
        self._state_item.setTitle_(f"Gaze: {'webcam' if source == 'camera' else 'pointer'}")

    def _on_gaze_point(self, x: float, y: float) -> None:
        snapper = self.snapper
        if snapper is None:
            return
        shown = (snapper.update(time.monotonic(), x, y), snapper.pinned)
        if shown != self._highlighted:
            self._highlighted = shown
            _on_main(self.overlay.highlight, *shown)

    def _on_wink(self, started: float) -> None:
        """A wink clicks whatever is highlighted, or was just before the eye shut."""
        snapper = self.snapper
        if snapper is None or self.sleeping or self._calibrator is not None:
            return
        element = snapper.pinned_element() if snapper.pinned else None
        element = element or snapper.element_during(started - WINK_LOOKBACK_S, started)
        if element is None:
            log.info("wink: nothing highlighted to click")
            return
        threading.Thread(target=self._wink_click, args=(element,), daemon=True).start()

    def _wink_click(self, element) -> None:
        if self.config.sounds:
            _on_main(_play, "Morse")
        try:
            log.info("wink: %s", inp.click(element))
        except Exception as err:
            log.warning("wink click failed: %s", err)

    def switch_gaze(self, source: str) -> None:
        self._start_gaze(source)
        if source == "camera" and not self.gaze.calibrated:
            self.start_calibration()

    def toggle_highlight(self) -> None:
        on = self._highlight_item.state() != 1
        self._highlight_item.setState_(1 if on else 0)
        self.overlay.set_highlight_enabled(on)
        if on:
            self.overlay.highlight(*self._highlighted)

    def gaze_error_px(self) -> float:
        if isinstance(self.gaze, MouseGaze):
            return MOUSE_ERROR_PX
        if isinstance(self.gaze, CameraGaze) and self.gaze.model is not None:
            return self.gaze.model.cv_error_px
        return UNCALIBRATED_ERROR_PX

    def start_calibration(self, quick: bool = False) -> None:
        if not isinstance(self.gaze, CameraGaze):
            self.speaker.say("Calibration is only for webcam eye tracking.")
            return
        if self.gaze.error:
            self._show("error", f"Camera problem: {self.gaze.error}")
            return
        if self._calibrator is not None:
            return
        quick = quick and self.gaze.calibrated
        self.overlay.set_highlight_enabled(False)
        self.speaker.say(
            "Look at the dot." if quick else "Let's calibrate. Look at each dot until it moves."
        )
        self._calibrator = Calibrator(self.gaze, self._calibrated, self.speaker.say, quick)
        self._calibrator.start()

    def _calibrated(self, model, message: str) -> None:
        self._calibrator = None
        if model is not None:
            self.gaze.set_model(model)
        self.overlay.set_highlight_enabled(self._highlight_item.state() == 1)
        self._show("idle" if model else "error", message)
        self._hide_later(5.0)
        self.speaker.say(message)

    # status

    def _show(self, state: str, message: str) -> None:
        self._status_stamp += 1
        if threading.current_thread() is threading.main_thread():
            self.overlay.status(state, message)
        else:
            _on_main(self.overlay.status, state, message)

    def _hide_later(self, seconds: float) -> None:
        """Hide the status pill unless something else is shown before then."""
        stamp = self._status_stamp

        def hide():
            if self._status_stamp == stamp and not self._conversing:
                self.overlay.status("idle", "")

        AppHelper.callLater(seconds, hide)

    def _status_listening(self, status: str) -> None:
        if status == "listening":
            self._pause_item.setTitle_("Pause Listening")
        else:
            self._show("error", status)

    def _on_speaking(self, speaking: bool) -> None:
        if self.listener is not None and not self.config.barge_in:
            (self.listener.muted.set if speaking else self.listener.muted.clear)()
        if not speaking:
            self._settle()

    def set_sleeping(self, sleeping: bool, goodbye: str = "") -> None:
        self.sleeping = sleeping
        self._pause_item.setTitle_("Resume Listening" if sleeping else "Pause Listening")
        if sleeping:
            self._stop_everything(STOPPED)
            self._end_conversation(sound=False)
            self._unpin()
            self.speaker.say(goodbye)
            self._show("idle", "Sleeping. Say “Peek, wake up.”")
        else:
            self._show("listening", "Listening.")
        self._hide_later(3.0)

    # conversation
    #
    # Saying "Peek" opens a conversation. While it's open, everything the user
    # says is for Peek, no wake word needed. It stays open while Peek is
    # working, speaking or waiting for an answer, and while the user is
    # talking, then closes after a quiet spell.

    def _open_conversation(self) -> None:
        if not self._conversing and self.config.sounds:
            _on_main(_play, "Tink")
        self._conversing = True
        self._conversation_deadline = time.monotonic() + self.config.conversation_s

    def _end_conversation(self, sound: bool = True) -> None:
        if not self._conversing:
            return
        self._conversing = False
        self._conversation_ended = time.monotonic()
        if self._held_for_request:
            self._held_for_request = False
            self._unpin_if_idle()
        self._last_reply = ""
        if sound and self.config.sounds:
            _on_main(_play, "Pop")
        self._show("idle", "")

    def _working(self) -> bool:
        with self._jobs_lock:
            return bool(self._jobs)

    def _engaged(self) -> bool:
        """Peek is mid-exchange: working, talking, waiting on an answer, or being talked to."""
        user_talking = self.listener is not None and self.listener.turn_started is not None
        return self._working() or self._answers is not None or self.speaker.speaking or user_talking

    def _settle(self) -> None:
        """Peek just went quiet: give the user time to reply before closing."""
        if not self._conversing or self._answers is not None or self.speaker.speaking:
            return
        if self._working():
            return
        last = self.speaker.last_spoken()
        asked = last is not None and last.text.rstrip().endswith("?")
        wait = QUESTION_S if asked else self.config.conversation_s
        self._conversation_deadline = time.monotonic() + wait
        if self.listener is None or self.listener.turn_started is None:
            self._show("listening", self._last_reply or "Listening…")

    def _tick(self) -> None:
        if (
            self._conversing
            and time.monotonic() > self._conversation_deadline
            and not self._engaged()
        ):
            self._end_conversation()
        AppHelper.callLater(0.5, self._tick)

    def _for_peek(self, started: float) -> bool:
        """Whether speech without the wake word, begun at `started`, is meant for Peek."""
        if not self.config.require_wake_word:
            return True
        return self._conversing or started <= self._conversation_ended

    def _stop_everything(self, reason: str) -> None:
        with self._jobs_lock:
            for job in self._jobs:
                job.cancel_reason = job.cancel_reason or reason
        answers = self._answers
        if answers is not None:
            answers.put(None)
        self.speaker.interrupt()

    def _submit(self, job: Job, held: bool = False) -> None:
        """Run a command. A new request interrupts whatever Peek was doing."""
        self._stop_everything(REPLACED)
        snapper = self.snapper
        if snapper is not None and not ((job.said_name or held) and snapper.pinned):
            # Follow-ups (and a name the partials missed) pin what they looked at while speaking.
            snapper.pin(job.started - GAZE_LEAD_S, job.ended)
        with self._jobs_lock:
            self._jobs.append(job)
        self._commands.put(job)

    # pinning
    #
    # When the user says "Peek", the highlight pins to what they're looking at
    # and stays there until the request is done, so they can look away while
    # they finish speaking and while Peek works on it. Saying just "Peek" and
    # pausing keeps the pin for the request that follows. It lets go while Peek asks them something, so they can
    # look at their answer, then pins what they looked at.

    def _pin(self, start: float | None = None, end: float | None = None) -> None:
        if self.snapper is not None:
            self.snapper.pin(start, end)
            log.debug("pinned (%s, %s)", start, end)

    def _unpin(self) -> None:
        self._held_for_request = False
        if self.snapper is not None and self.snapper.pinned:
            self.snapper.unpin()
            log.debug("unpinned")

    def _unpin_if_idle(self) -> None:
        if self._held_for_request:
            return
        user_talking = self.listener is not None and self.listener.turn_started is not None
        if not self._working() and self._answers is None and not user_talking:
            self._unpin()

    # voice

    def _without_own_voice(self, transcript: str, started: float, ended: float) -> str:
        if not self.config.barge_in:
            return transcript.strip()
        return strip_echo(transcript, self.speaker.spoken_between(started, ended))

    def _on_partial(self, text: str) -> None:
        if self.sleeping or self.listener is None:
            return
        now = time.monotonic()
        heard = self._without_own_voice(text, self.listener.turn_started or now, now)
        if not heard:
            return
        command = wake.split_command(heard, self.config.wake_words)
        for_peek = command is not None or self._for_peek(self.listener.turn_started or now)
        if not for_peek:
            return
        if command is not None and not self._name_heard:
            self._name_heard = True
            if self._answers is None and not self._held_for_request:
                # What they looked at since they started talking, not where
                # their eyes are by the time the transcript catches up.
                started = self.listener.turn_started or now
                self._pin(started - GAZE_LEAD_S, now)
        if self.speaker.speaking:
            if not self._wants_the_floor(heard, command):
                return
            self.speaker.interrupt()
        self._show("listening", (heard if command is None else command) or "Listening…")

    def _wants_the_floor(self, heard: str, command: str | None) -> bool:
        """Whether the user is talking over Peek to say something, not just "mm-hmm"."""
        if command is not None or STOP.match(heard):
            return True
        return content_words(heard) >= (1 if self._answers is not None else 2)

    def _on_turn(self, turn: Turn) -> None:
        self._name_heard = False
        try:
            self._handle_turn(turn)
        finally:
            self._unpin_if_idle()

    def _handle_turn(self, turn: Turn) -> None:
        log.info("heard: %s", turn.transcript)
        heard = self._without_own_voice(turn.transcript, turn.started, turn.ended)
        if heard != turn.transcript.strip():
            log.info("without Peek's own voice: %r", heard)
        if not heard:
            return
        command = wake.split_command(heard, self.config.wake_words)
        said_name = command is not None
        if self.sleeping:
            if said_name and WAKE.match(command):
                _on_main(self.set_sleeping, False)
                self._open_conversation()
                self.speaker.say("I'm listening.")
            return
        if command is None:
            if not self._for_peek(turn.started):
                return
            command = heard
            last = self.speaker.last_spoken()
            asked = last is not None and last.text.rstrip().endswith("?")
            if is_backchannel(command) and self._answers is None and not asked:
                return
        held, self._held_for_request = self._held_for_request, False
        self._open_conversation()
        if not command:
            # Just the name: stop talking if need be and listen, keeping the
            # pin on what they looked at for the request that follows.
            if self._answers is None:
                if self.snapper is not None and not self.snapper.pinned:
                    self._pin(turn.started - GAZE_LEAD_S, turn.ended)
                self._held_for_request = True
            self.speaker.interrupt()
            self._show("listening", "Listening…")
            return
        if STOP.match(command):
            self._stop_everything(STOPPED)
            self._show("listening", "Okay.")
            return
        if DISMISS.match(command) and self._answers is None and not self._working():
            self.speaker.interrupt()
            self._end_conversation(sound=False)
            self.speaker.say(random.choice(ACKNOWLEDGE))
            return
        if SLEEP.match(command):
            _on_main(self.set_sleeping, True, "Okay, I'll wait.")
            return
        if CALIBRATE.match(command):
            _on_main(self.start_calibration)
            return
        if RECENTER.match(command):
            _on_main(self.start_calibration, True)
            return
        answers = self._answers
        if answers is not None:
            log.info("answer: %s", command)
            answers.put(Turn(command, turn.started, turn.ended))
            return
        log.info("command: %s", command)
        self._submit(Job(command, turn.started, turn.ended, said_name), held)

    # commands

    def _snapshot(self, start: float, end: float) -> screen_context.Snapshot:
        fixations = self.history.fixations(start - GAZE_LEAD_S, end)
        main = primary(fixations)
        if main is None:
            latest = self.history.latest()
            if latest is not None and time.monotonic() - latest[0] < 2.0:
                main = Fixation(latest[1], latest[2], latest[0], latest[0])
        others = [f for f in fixations if f is not main][-4:]
        snapper = self.snapper
        highlighted = None
        if snapper is not None:
            highlighted = snapper.pinned_element() if snapper.pinned else None
            highlighted = highlighted or snapper.element_during(start - GAZE_LEAD_S, end)
        return screen_context.build(
            main,
            others,
            self.gaze_error_px(),
            want_image=self.config.send_screenshot,
            highlighted=highlighted,
        )

    def _refresh(self) -> screen_context.Snapshot:
        now = time.monotonic()
        return self._snapshot(now - 0.8, now)

    def _say(self, text: str) -> None:
        self._show("speaking", text)
        self.speaker.say(text)

    def _ask(self, question: str) -> Answer | None:
        """Speak `question` and wait for the user's answer (None if none came)."""
        answers: queue.Queue[Turn | None] = queue.Queue()
        self._answers = answers
        self._unpin()
        try:
            self._say(question)
            self.speaker.wait()
            if answers.empty():
                self._show("listening", question)
            deadline = time.monotonic() + ASK_TIMEOUT_S
            while True:
                try:
                    turn = answers.get(timeout=0.25)
                except queue.Empty:
                    talking = self.listener is not None and self.listener.turn_started is not None
                    if time.monotonic() > deadline and not talking:
                        return None
                    continue
                if turn is None:
                    return None
                self._show("working", turn.transcript)
                start = turn.started - GAZE_LEAD_S
                self._pin(start, turn.ended)
                main = primary(self.history.fixations(start, turn.ended))
                snapper = self.snapper
                return Answer(
                    turn.transcript,
                    snapper.element_during(start, turn.ended) if snapper else None,
                    (main.x, main.y) if main else None,
                )
        finally:
            self._answers = None

    def _worker(self) -> None:
        while True:
            job = self._commands.get()
            try:
                reply = self._run(job)
            finally:
                with self._jobs_lock:
                    self._jobs.remove(job)
                self._unpin_if_idle()
            log.info("reply: %r", reply)
            if reply is None:
                continue
            if reply == SILENT:
                if not self._working():
                    self._end_conversation()
                continue
            if reply:
                self._last_reply = reply
                self._say(reply)
            self._settle()

    def _run(self, job: Job) -> str | None:
        if job.cancel_reason:
            return None
        said = job.command
        self._show("working", said)
        try:
            return self.agent.handle(
                said,
                self._snapshot(job.started, job.ended),
                refresh=self._refresh,
                recalibrate=lambda quick=False: _on_main(self.start_calibration, quick),
                said_name=job.said_name,
                ask=self._ask,
                say=self._say,
                on_step=lambda step: self._show("working", f"{said} — {step}"),
                cancelled=lambda: job.cancel_reason,
            )
        except Exception as err:
            log.exception("command failed")
            self._show("error", str(err))
            return "Sorry, something went wrong on my end."

    def quit(self) -> None:
        if self.gaze is not None:
            self.gaze.stop()
        if self.snapper is not None:
            self.snapper.stop()
        if self.listener is not None:
            self.listener.stop()
        sys.stdout.flush()
        AppKit.NSApp.terminate_(None)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    GazeApp(load_config()).run()
