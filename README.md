# Peek

A hands-free assistant for people who find a keyboard and mouse hard to use.
Look at something on your Mac, say what you want, and Peek does it.

- "Peek, make this full screen." (looking at a window)
- "Peek, close that."
- "Peek, put this on the left and that on the right." (look at one window, then the other)
- "Peek, write 'running ten minutes late' here." (looking at a text box)
- "Peek, what does this say?" / "read this to me"
- "Peek, scroll down." / "open Safari" / "go to sleep" / "wake up" / "recalibrate" / "re-center"
- Wink one eye to click whatever is outlined in blue, no words needed (webcam
  only; turn off with `PEEK_WINK_CLICK=false`).

### Talking to Peek

Say "Peek" once and you're in a conversation: keep talking without the wake
word ("now the other one", "undo that", "a bit more"). You hear a soft tink
when Peek starts listening and a pop when it stops, about 15 seconds after the
last exchange (30 if Peek just asked you something). "Thanks" or "that's all"
ends it right away.

- **Interrupt any time.** Talk over Peek and it stops speaking. "Stop", "wait"
  or "never mind" also cancels whatever it's doing. A new request while it's
  working replaces the old one, and Peek works out whether you corrected it or
  added to it. "Okay" and "mm-hmm" don't interrupt.
- **It waits for answers.** When Peek needs something ("Which window, the
  left or the right?", "Close it without saving?"), it asks and waits for your
  reply, then finishes the job. You can look at the thing you mean while you
  answer.
- **It remembers the last few minutes**, including what it did, so follow-ups
  and corrections work.
- **Side conversations are usually ignored.** If you talk to someone else
  mid-conversation, Peek should stay quiet and stop listening.
- For slower jobs it gives a quick heads-up ("Let me find that") instead of
  going silent.

The mic stays on while Peek talks so you can interrupt. Peek strips its own
voice out of what the mic hears, but headphones work best. In a loud room,
or if Peek keeps cutting itself off, set `PEEK_BARGE_IN=false`. Peek then
can't hear you while it's talking, so you have to wait for it to finish.

## How it works

| Piece | What it uses |
| --- | --- |
| Eye tracking | Webcam + MediaPipe Face Landmarker (iris, eyelid blendshapes, head pose), a 25-dot calibration plus 9 more after a head shift, ridge regression with the feature set picked per calibration, a One Euro filter, and a blue outline around the control or text you're looking at. Or the pointer, for head-mouse and hardware eye-tracker users. |
| Listening | [Cartesia](https://www.cartesia.ai/) Ink-2 realtime STT with built-in turn detection, so there's no button to hold. |
| Speaking | Cartesia Sonic 3.6, streamed. Falls back to macOS `say`. |
| Context | macOS Accessibility: the window and element under your gaze, interactive elements near it (webcam gaze is off by 100-200 px), other places you looked while speaking, all windows, and an optional screenshot crop with your gaze marked. |
| Acting | Claude with tools for windows (full screen, close, minimize, tile, split), typing, clicking, keys, scrolling, opening apps and links. |

## Setup

```bash
uv sync
cp .env.example .env   # add CARTESIA_API_KEY and ANTHROPIC_API_KEY
uv run peek
```

`models/face_landmarker.task` is MediaPipe's public Face Landmarker model
(`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`).

macOS asks for permissions for the app you launch from (Terminal, Cursor, ...).
Grant them in System Settings → Privacy & Security:

1. **Accessibility**: reading the screen and acting on it. Restart Peek after granting it.
2. **Camera**: eye tracking.
3. **Microphone**: listening.
4. **Screen Recording** (optional): the screenshot crop around your gaze.

On first launch with the webcam, calibration starts by itself: look at each
dot until it moves, and shift your head a little when asked. It takes about a
minute. Esc cancels. Calibration is saved to
`~/Library/Application Support/Peek/calibration.npz`, and the raw samples
next to it in `calibration_samples.npz`.

While you look around, a blue outline settles on the button, link, field or
text you're looking at, in any app or web page that supports Accessibility.
That's what "this" means in a command. When you say "Peek", the outline locks onto that
thing and turns bolder, and stays there until Peek is done, so you can look
away while it works. It lets go while Peek asks you a question, so you can look
at your answer. If the outline keeps landing next to
what you're looking at (you moved your chair, the light changed), say
"Peek, re-center" and look at a single dot. That fixes a steady offset in
two seconds. Recalibrate fully if it's off in different ways in different
places.

Peek lives in the menu bar (the eye icon), with Pause, Calibrate Eyes,
Re-center Eyes, Show Highlight, Gaze Source and Quit.

## Settings

All optional, in `.env`. See `.env.example`.

| Name | Default | |
| --- | --- | --- |
| `PEEK_SOURCE` | `camera` | `mouse` uses the pointer as your gaze |
| `PEEK_REQUIRE_WAKE_WORD` | `true` | `false` treats every sentence as a command |
| `PEEK_WAKE_WORDS` | `peek,hey computer` | matched loosely at the start of a sentence or clause |
| `PEEK_MODEL` | `claude-sonnet-5-5` | `claude-haiku-5-5` is faster |
| `PEEK_SEND_SCREENSHOT` | `true` | |
| `PEEK_HIGHLIGHT` | `true` | outline the control or text you're looking at |
| `PEEK_NOISE_GATE_DB` | `12` | how far above the room's background level speech must be to count; raise it in noisy rooms, `0` turns it off |
| `PEEK_BARGE_IN` | `true` | keep the mic on while Peek talks so you can interrupt it |
| `PEEK_CONVERSATION_S` | `15` | how long Peek keeps listening without the wake word after an exchange |
| `PEEK_SOUNDS` | `true` | the tink and pop when a conversation starts and ends |
| `PEEK_VOICE_ID` | a Cartesia stock voice | |

## Code

```
peek/
  app.py             menu bar app, conversation and interruption handling, command worker
  wake.py            fuzzy wake word matching
  gaze/              features, calibration model, filter, fixations, camera/pointer sources
  voice/             Cartesia listener (turn detection), speaker, noise gate, echo removal
  screen/            Accessibility helpers, windows/screens, context snapshot
  actions/           window actions and synthetic input
  agent/             Claude tool loop and tool definitions
  ui/                overlays (highlight, status pill) and calibration
```

`uv run pytest` runs the tests. `uv run ruff check` lints.
