# Deixis

A hands-free assistant for people who find a keyboard and mouse hard to use.
Look at something on your Mac, say what you want, and Deixis does it.

- "Deixis, make this full screen." (looking at a window)
- "Deixis, close that."
- "Deixis, put this on the left and that on the right." (look at one window, then the other)
- "Deixis, write 'running ten minutes late' here." (looking at a text box)
- "Deixis, what does this say?" / "read this to me"
- "Deixis, scroll down." / "open Safari" / "go to sleep" / "wake up" / "recalibrate"

Saying only "Deixis" gets a "Yes?", and you can then answer without the wake word.

## How it works

| Piece | What it uses |
| --- | --- |
| Gaze | Webcam + MediaPipe Face Landmarker (iris, eyelid blendshapes, head pose), a 13-dot calibration, ridge regression, a One Euro filter. Or the pointer, for head-mouse and hardware eye-tracker users. |
| Listening | [Cartesia](https://www.cartesia.ai/) Ink-2 realtime STT with built-in turn detection, so there's no button to hold. |
| Speaking | Cartesia Sonic 3.6, streamed. Falls back to macOS `say`. |
| Context | macOS Accessibility: the window and element under your gaze, interactive elements near it (webcam gaze is off by 100-200 px), other places you looked while speaking, all windows, and an optional screenshot crop with your gaze marked. |
| Acting | Claude with tools for windows (full screen, close, minimize, tile, split), typing, clicking, keys, scrolling, opening apps and links. |

## Setup

```bash
uv sync
cp .env.example .env   # add CARTESIA_API_KEY and ANTHROPIC_API_KEY
uv run deixis
```

`models/face_landmarker.task` is MediaPipe's public Face Landmarker model
(`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`).

macOS asks for permissions for the app you launch from (Terminal, Cursor, ...).
Grant them in System Settings → Privacy & Security:

1. **Accessibility**: reading the screen and acting on it. Restart Deixis after granting it.
2. **Camera**: eye tracking.
3. **Microphone**: listening.
4. **Screen Recording** (optional): the screenshot crop around your gaze.

On first launch with the webcam, calibration starts by itself: look at each
dot until it moves. Esc cancels. Calibration is saved to
`~/Library/Application Support/Deixis/calibration.npz`.

Deixis lives in the menu bar (the eye icon), with Pause, Calibrate Eyes, Show
Gaze Dot, Gaze Source and Quit.

## Settings

All optional, in `.env`. See `.env.example`.

| Name | Default | |
| --- | --- | --- |
| `DEIXIS_GAZE` | `camera` | `mouse` uses the pointer as your gaze |
| `DEIXIS_REQUIRE_WAKE_WORD` | `true` | `false` treats every sentence as a command |
| `DEIXIS_WAKE_WORDS` | `deixis,dixie,dexis,daxis,hey computer` | matched loosely near the start |
| `DEIXIS_MODEL` | `claude-sonnet-5-5` | `claude-haiku-5-5` is faster |
| `DEIXIS_SEND_SCREENSHOT` | `true` | |
| `DEIXIS_VOICE_ID` | a Cartesia stock voice | |

## Code

```
deixis/
  app.py             menu bar app, wake word routing, command worker
  wake.py            fuzzy wake word matching
  gaze/              features, calibration model, filter, fixations, camera/pointer sources
  voice/             Cartesia listener (turn detection) and speaker
  screen/            Accessibility helpers, windows/screens, context snapshot
  actions/           window actions and synthetic input
  agent/             Claude tool loop and tool definitions
  ui/                overlays (gaze dot, status pill) and calibration
```

`uv run pytest` runs the tests. `uv run ruff check` lints.
