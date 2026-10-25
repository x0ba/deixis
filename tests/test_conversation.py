import queue
import threading
import time
from types import SimpleNamespace

import pytest

from peek import app as app_mod
from peek.app import DISMISS, STOP, GazeApp
from peek.config import Config
from peek.voice.listener import Turn
from peek.voice.speaker import ECHO_TAIL_S, Speaker, Utterance


@pytest.mark.parametrize(
    "text",
    ["stop", "Stop.", "wait", "hold on a sec", "never mind", "okay stop", "no, stop it", "shh"],
)
def test_stop_phrases(text):
    assert STOP.match(text)


@pytest.mark.parametrize("text", ["stop the music", "wait for the page to load", "cancel my 3pm"])
def test_requests_that_start_like_stop_phrases(text):
    assert not STOP.match(text)


@pytest.mark.parametrize("text", ["thanks", "Thank you so much.", "that's all for now", "bye Peek"])
def test_dismissals(text):
    assert DISMISS.match(text)


def test_dismissal_needs_to_be_the_whole_turn():
    assert not DISMISS.match("thanks, now close it")


def test_spoken_between_covers_the_echo_tail():
    speaker = Speaker.__new__(Speaker)
    speaker._lock = threading.Lock()
    now = time.monotonic()
    speaker._history = [
        Utterance("long ago", now - 30, now - 29),
        Utterance("just said", now - 3, now - 0.5),
        Utterance("saying now", now - 0.2),
    ]
    assert speaker.spoken_between(now - 0.5 + ECHO_TAIL_S / 2, now) == ["just said", "saying now"]
    assert speaker.spoken_between(now + ECHO_TAIL_S, now + 2) == ["saying now"]


class FakeSpeaker:
    def __init__(self):
        self.said: list[str] = []
        self.interrupts = 0
        self.speaking = False
        self.history: list[Utterance] = []

    def say(self, text):
        if text:
            self.said.append(text)

    def interrupt(self):
        self.interrupts += 1
        self.speaking = False

    def spoken_between(self, start, end):
        return [u.text for u in self.history]

    def last_spoken(self):
        return self.history[-1] if self.history else None

    def talk(self, text):
        self.history.append(Utterance(text, time.monotonic()))
        self.speaking = True


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(app_mod, "_on_main", lambda fn, *args: None)
    config = Config("", "", "model", "voice", "mouse", wake_words=["peek"], sounds=False)
    app = GazeApp(config)
    app.overlay = SimpleNamespace(status=lambda state, message: None)
    app.speaker = FakeSpeaker()
    app.listener = SimpleNamespace(turn_started=None)
    return app


def hear(app, text):
    now = time.monotonic()
    app._on_turn(Turn(text, now, now + 1))


def submitted(app) -> list[str]:
    return [job.command for job in app._jobs]


def test_needs_the_wake_word_to_start(app):
    hear(app, "close this window")
    assert submitted(app) == []
    hear(app, "Peek, close this window")
    assert submitted(app) == ["close this window"]


def test_follow_ups_need_no_wake_word(app):
    hear(app, "Peek")
    hear(app, "make it full screen")
    assert submitted(app) == ["make it full screen"]
    assert not app._jobs[0].said_name


def test_conversation_closes_after_a_quiet_spell(app, monkeypatch):
    monkeypatch.setattr(app_mod.AppHelper, "callLater", lambda *a: None)
    hear(app, "Peek")
    app._conversation_deadline = time.monotonic() - 1
    app._tick()
    hear(app, "make it full screen")
    assert submitted(app) == []


def test_new_request_interrupts_the_current_one(app):
    hear(app, "Peek, read this to me")
    hear(app, "actually close it")
    first, second = app._jobs
    assert first.cancel_reason == app_mod.REPLACED
    assert second.cancel_reason is None


def test_stop_cancels_without_a_new_request(app):
    hear(app, "Peek, scroll down slowly")
    hear(app, "stop")
    assert app._jobs[0].cancel_reason == app_mod.STOPPED
    assert len(app._jobs) == 1
    assert app.speaker.interrupts >= 1


def test_answers_go_to_the_waiting_question(app):
    hear(app, "Peek")
    answers = app._answers = queue.Queue()
    hear(app, "the left one")
    assert answers.get_nowait().transcript == "the left one"
    assert submitted(app) == []


def test_own_voice_is_ignored_but_reply_after_it_is_heard(app):
    hear(app, "Peek")
    app.speaker.talk("Do you want me to close Safari?")
    hear(app, "Do you want me to close Safari?")
    assert submitted(app) == []
    hear(app, "Do you want me to close Safari? Yes please.")
    assert submitted(app) == ["Yes please."]


def test_backchannel_while_peek_talks_is_ignored(app):
    hear(app, "Peek")
    app.speaker.talk("Here's what the article says.")
    hear(app, "okay")
    assert submitted(app) == []


def test_talking_over_peek_interrupts_it(app):
    hear(app, "Peek")
    app.speaker.talk("Here is a long article about the history of typefaces.")
    app.speaker.interrupts = 0
    app.listener.turn_started = time.monotonic()
    app._on_partial("Here is a long article. Okay")
    assert app.speaker.interrupts == 0
    app._on_partial("Here is a long article. Skip to the end")
    assert app.speaker.interrupts == 1


def test_thanks_ends_the_conversation(app):
    hear(app, "Peek")
    hear(app, "thanks")
    assert not app._conversing
    assert app.speaker.said and app.speaker.said[-1] in app_mod.ACKNOWLEDGE
    hear(app, "close it")
    assert submitted(app) == []


class FakeSnapper:
    def __init__(self):
        self.pinned = False
        self.pins: list[tuple] = []

    def pin(self, start=None, end=None):
        self.pinned = True
        self.pins.append((start, end))

    def unpin(self):
        self.pinned = False


def test_saying_the_name_pins_what_they_look_at_until_the_request_is_done(app):
    app.snapper = FakeSnapper()
    app.listener.turn_started = time.monotonic()
    app._on_partial("Peek, close")
    assert app.snapper.pins == [(None, None)]
    app._on_partial("Peek, close this")
    assert len(app.snapper.pins) == 1
    app.listener.turn_started = None
    hear(app, "Peek, close this")
    assert app.snapper.pinned and len(app.snapper.pins) == 1
    app._jobs.clear()
    app._unpin_if_idle()
    assert not app.snapper.pinned


def test_just_the_name_lets_go_of_the_pin(app):
    app.snapper = FakeSnapper()
    app.listener.turn_started = time.monotonic()
    app._on_partial("Peek")
    app.listener.turn_started = None
    hear(app, "Peek")
    assert app.snapper.pins and not app.snapper.pinned


def test_follow_ups_pin_what_they_looked_at_while_speaking(app):
    app.snapper = FakeSnapper()
    hear(app, "Peek")
    hear(app, "make it full screen")
    assert app.snapper.pinned
    start, end = app.snapper.pins[-1]
    assert start is not None and end is not None
