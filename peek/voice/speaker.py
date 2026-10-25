"""Spoken replies with Cartesia Sonic, streamed straight to the speakers.

Falls back to macOS `say` when there is no Cartesia key or the request fails,
so the user always hears an answer. Speech stops the moment `interrupt()` is
called, and the speaker remembers what it said and when, so the listener can
tell its own voice from the user's.
"""

import logging
import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import sounddevice as sd
from cartesia import Cartesia

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100
MODEL = "sonic-3.6"
SLICE_BYTES = SAMPLE_RATE // 20 * 2  # write 50 ms at a time so interrupts land quickly
ECHO_TAIL_S = 1.0  # speech recognition lags the speakers by about this much


@dataclass
class Utterance:
    text: str
    started: float
    ended: float | None = None


class Speaker:
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        on_speaking: Callable[[bool], None] = lambda _: None,
    ):
        self._client = Cartesia(api_key=api_key) if api_key else None
        self._voice_id = voice_id
        self._on_speaking = on_speaking
        self._queue: queue.Queue[tuple[int, str]] = queue.Queue()
        self._ws = None
        self._lock = threading.Lock()
        self._generation = 0
        self._pending = 0
        self._idle = threading.Event()
        self._idle.set()
        self._history: deque[Utterance] = deque(maxlen=12)
        self._say_process: subprocess.Popen | None = None
        threading.Thread(target=self._run, name="speaker", daemon=True).start()

    @property
    def speaking(self) -> bool:
        return not self._idle.is_set()

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._pending += 1
            self._idle.clear()
            self._queue.put((self._generation, text))

    def interrupt(self) -> None:
        """Stop talking now and drop anything queued."""
        with self._lock:
            self._generation += 1
            process = self._say_process
        if process is not None:
            process.terminate()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been said or interrupted."""
        return self._idle.wait(timeout)

    def spoken_between(self, start: float, end: float) -> list[str]:
        """What Peek said that the mic could have heard between `start` and `end`."""
        now = time.monotonic()
        with self._lock:
            return [
                u.text
                for u in self._history
                if u.started <= end and (u.ended or now) + ECHO_TAIL_S >= start
            ]

    def last_spoken(self) -> Utterance | None:
        with self._lock:
            return self._history[-1] if self._history else None

    def _current(self, generation: int) -> bool:
        return generation == self._generation

    def _run(self) -> None:
        while True:
            generation, text = self._queue.get()
            try:
                if self._current(generation):
                    self._speak(generation, text)
            finally:
                with self._lock:
                    self._pending -= 1
                    done = self._pending == 0
                    if done:
                        self._idle.set()
                if done:
                    self._on_speaking(False)

    def _speak(self, generation: int, text: str) -> None:
        utterance = Utterance(text, time.monotonic())
        with self._lock:
            self._history.append(utterance)
        self._on_speaking(True)
        try:
            if self._client is None:
                raise RuntimeError("no Cartesia key")
            self._stream(generation, text)
        except Exception as err:
            log.warning("Cartesia TTS failed (%s); using macOS say", err)
            self._ws = None
            self._say(generation, text)
        finally:
            utterance.ended = time.monotonic()

    def _say(self, generation: int, text: str) -> None:
        with self._lock:
            if not self._current(generation):
                return
            self._say_process = subprocess.Popen(["say", text])
        try:
            self._say_process.wait()
        finally:
            with self._lock:
                self._say_process = None

    def _stream(self, generation: int, text: str) -> None:
        if self._ws is None:
            self._ws = self._client.tts.websocket_connect().__enter__()
        ctx = self._ws.context(
            model_id=MODEL,
            voice=self._voice_id,
            output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE},
            language="en",
        )
        ctx.push(text)
        ctx.no_more_inputs()
        with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as out:
            for response in ctx.receive():
                if response.type == "error":
                    raise RuntimeError(response.message or response.title)
                if response.type != "chunk" or not response.audio:
                    continue
                for i in range(0, len(response.audio), SLICE_BYTES):
                    if not self._current(generation):
                        ctx.cancel()
                        out.abort()
                        return
                    out.write(response.audio[i : i + SLICE_BYTES])
