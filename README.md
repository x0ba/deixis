# Peek

A hands-free assistant for people who find a keyboard and mouse hard to use.
Look at something on your Mac, say what you want, and Peek does it.

- "Peek, make this full screen." (looking at a window)
- "Peek, close that."
- "Peek, put this on the left and that on the right." (look at one window, then the other)
- "Peek, write 'running ten minutes late' here." (looking at a text box)
- "Peek, what does this say?" / "read this to me"
- "Peek, scroll down." / "open Safari" / "go to sleep" / "wake up" / "recalibrate" / "quick calibrate" / "re-center"
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
| Eye tracking | Webcam + MediaPipe Face Landmarker (iris, eyelid blendshapes, head pose), a 25-dot calibration plus 9 more after a head shift, a moving dot to follow, and 5 check dots that measure the real accuracy; ridge regression with the feature set picked per calibration; a One Euro filter; drift correction from your clicks; and a blue outline that goes to the control or text that has gathered the most gaze over the last fraction of a second, and only moves when another one clearly wins. Or the pointer, for head-mouse and hardware eye-tracker users. |
| Listening | [Cartesia](https://www.cartesia.ai/) Ink-2 realtime STT with built-in turn detection, so there's no button to hold. |
| Speaking | Cartesia Sonic 3.6, streamed. Falls back to macOS `say`. |
| Context | macOS Accessibility: the window and element under your gaze, the buttons and links of the post, message or row it's part of, interactive elements near it (webcam gaze is off by 100-200 px), other places you looked while speaking, all windows, and an optional screenshot crop with your gaze marked. |
| Acting | Claude with tools for windows (full screen, close, minimize, tile, split), typing, clicking, keys, scrolling, opening apps and links. |

### Cartesia features used

All through the Python SDK (`cartesia[websockets]` 4.2).

- **Ink-2 streaming speech-to-text** over the auto-finalize WebSocket
  (`stt.auto_finalize.websocket`). The mic streams 16 kHz 16-bit PCM
  continuously, so there's no push-to-talk.
- **Native turn detection.** Peek reacts to the turn events directly:
  - `turn.start` marks when you started talking. Peek uses it to look up what
    you were looking at then, not where your eyes ended up once the
    transcript arrived.
  - `turn.update` and `turn.eager_end` partial transcripts let Peek spot the
    wake word early, lock the outline, show what it's hearing in the pill,
    and stop talking as soon as you talk over it.
  - `turn.end` delivers the final transcript, which is what Peek acts on.
    `turn_end_timeout_ms` is set to 2500.
- **Keyterm boosting** (`keyterm`) for the wake word and command words the
  model tends to mishear: "Peek", "full screen", "split screen", "minimize",
  "text box", "re-center".
- **Sonic 3.6 streaming text-to-speech** over one persistent WebSocket, with
  a context per reply. Raw 44.1 kHz PCM chunks play as they arrive, written
  50 ms at a time so an interruption cuts the audio almost right away and
  cancels the context.
- **Stock voices**, picked with `PEEK_VOICE_ID`.

## Setup

### What you need

- A Mac with a webcam and a microphone. The built-in ones are fine;
  headphones help Peek tell your voice from its own.
- [uv](https://docs.astral.sh/uv/getting-started/installation/), which also
  installs the right Python for you: `brew install uv`, or
  `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Two API keys:
  - **Anthropic**, for understanding what you ask:
    [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
  - **Cartesia**, for listening and speaking:
    [play.cartesia.ai/keys](https://play.cartesia.ai/keys)

If you already use a head mouse or a hardware eye tracker that moves the
pointer, you don't need the webcam: set `PEEK_SOURCE=mouse` (see
[Settings](#settings)).

### Install and run

```bash
git clone https://github.com/x0ba/deixis.git
cd deixis
uv sync
cp .env.example .env
```

Open `.env` in any text editor and paste your keys after `CARTESIA_API_KEY=`
and `ANTHROPIC_API_KEY=`. Then start Peek:

```bash
uv run peek
```

Leave the terminal open: Peek runs as long as it does, and logs what it heard
and did there. To stop it, choose Quit Peek from the eye icon in the menu bar,
or press Ctrl+C in the terminal.

`models/face_landmarker.task` is MediaPipe's public Face Landmarker model
(`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`).

### Permissions

macOS asks for permissions for the app you launch from (Terminal, Cursor, ...),
not for Peek itself. Grant them in System Settings → Privacy & Security:

1. **Accessibility**: reading the screen and acting on it. Restart Peek after granting it.
2. **Camera**: eye tracking. If the camera doesn't come on after you allow it, restart Peek.
3. **Microphone**: listening.
4. **Screen Recording** (optional): the screenshot crop around your gaze.

If a permission prompt appears while the calibration dots are on screen, press
Esc to cancel calibration, answer the prompt, then choose Calibrate Eyes from
the menu bar.

### Calibration

On first launch with the webcam, calibration starts by itself: look at each
dot until it moves, shift your head a little when asked, then follow a dot as
it glides across the screen. A last few dots check how accurate the result
is. It takes about a minute and a half. Esc cancels. Calibration is saved to
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

When someone else sits down (the next person at a demo, say), they don't need
the full minute and a half: say "Peek, quick calibrate" or choose Quick
Calibrate Eyes. They look at 6 dots for about 12 seconds, and Peek adapts the
existing calibration to them, correcting its scale and offset. It's less
accurate than a full calibration, so it works best when the full one was done
well first. Clicking also helps: each time you click with the mouse or a wink,
Peek assumes you were looking at what you clicked and nudges the tracking a
little toward it.

Peek lives in the menu bar (the eye icon), with Pause, Calibrate Eyes, Quick
Calibrate Eyes, Re-center Eyes, Show Highlight, Gaze Source and Quit.

## Your first five minutes

1. Start Peek and follow the calibration dots. At the end Peek tells you its
   accuracy in pixels, measured on the check dots at the end; 100 to 200 is
   typical for a webcam.
2. Look around the screen. The blue outline follows your eyes. That's what
   Peek will act on when you say "this".
3. Look at a window and say **"Peek, make this full screen."** You hear a tink
   when Peek starts listening.
4. Without saying "Peek" again, say **"now exit full screen."** You're still in
   the conversation, so follow-ups don't need the wake word.
5. Say **"thanks."** The pop means Peek has stopped listening until you say
   "Peek" again.
6. When you want Peek to ignore you for a while, say **"Peek, go to sleep."**
   Say **"Peek, wake up"** to bring it back.

The pill at the top of the screen shows what Peek is doing:

| Dot | Meaning |
| --- | --- |
| Blue | listening |
| Amber | working on your request |
| Green | speaking |
| Red | something's wrong; the text says what |

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
| `PEEK_WINK_CLICK` | `true` | wink one eye to click what's outlined (webcam only) |
| `PEEK_LEARN_FROM_CLICKS` | `true` | when you click with the mouse or a wink, nudge the eye tracking toward what you clicked, so it stays accurate without re-centering |
| `PEEK_CAMERA` | `0` | which camera to use if you have more than one |
| `PEEK_VOICE_ID` | a Cartesia stock voice | |

Restart Peek after changing `.env`.

## Known limitations

- **macOS only, English only.** Peek speaks English, and its built-in
  commands ("stop", "go to sleep", "recalibrate", ...) are English phrases.
- **It needs the internet.** Listening has no offline fallback: if Cartesia
  can't be reached, Peek can't hear you and keeps trying to reconnect.
  Speaking falls back to macOS `say`, but understanding requests needs
  Anthropic.
- **The mic streams to Cartesia whenever Peek is running.** While Peek is
  paused, muted or below the noise gate it sends silence instead of what the
  mic hears, but the connection stays open.
- **Webcam gaze is only accurate to about 100-200 px.** In tightly packed
  lists, like a sidebar of 30 px rows, the outline sometimes lands on the
  item next to the one you mean. Quick calibration is less accurate than a
  full one.
- **Webcam eye tracking covers the main display only.** Calibration dots
  appear only on the main display. To work on other displays, use
  `PEEK_SOURCE=mouse` with a head mouse or eye tracker.
- **When the camera loses your face**, the outline stays where it was
  instead of disappearing.
- **Apps that draw their own content** (terminals, games, canvas-based apps)
  show Accessibility little or nothing, so the outline tends to stick to the
  last thing it found and Peek can't point at much inside them. Peek turns on full web Accessibility for Chrome, Brave, Edge,
  Arc and Vivaldi. Other Chromium-based browsers may not show it the page.
- **Telling its voice from yours works on the words, not the audio.** Peek
  removes from the transcript anything that matches what it just said. On
  speakers, if you repeat its words back while it's talking, those words can
  get dropped, and a loud room can still make it cut itself off. Headphones
  avoid both.
- **Short interjections don't interrupt.** While Peek is talking, it needs
  "stop", "wait", "never mind", the wake word, or at least two real words
  before it stops (one if it just asked you something). A single "no" isn't
  enough.
- **Turn detection decides when you're done.** A long pause mid-sentence can
  end your turn early, and Peek then acts on the first half.
- **The wake word is matched loosely** so that "Peak" and "Peek's" still
  count. A clause that starts with a similar word can wake it by mistake.
- **Side conversations are filtered by Claude, not by rules**, so sometimes
  Peek answers something you said to someone else.
- **Memory is short**: the last 8 exchanges from the last 10 minutes.
- **Wink-to-click can misfire** when you squint or rub an eye.

## Troubleshooting

The terminal Peek runs in logs every sentence it heard (`heard: ...`) and what
it did about it, which is usually the quickest way to see what went wrong.

**Peek doesn't respond when I talk.**
- Check the pill: if it says "Speech connection dropped", your
  `CARTESIA_API_KEY` is wrong or you're offline.
- Make sure the app you launched Peek from has Microphone access.
- Peek may be asleep. Say "Peek, wake up", or choose Resume Listening from the
  menu bar.
- If the log shows nothing while you talk, the noise gate may be cutting you
  off. Lower `PEEK_NOISE_GATE_DB`, or set it to `0`.

**Peek says "Sorry, something went wrong on my end."** Check
`ANTHROPIC_API_KEY` and, if you set it, `PEEK_MODEL`. The terminal shows the
actual error.

**Calibration says it couldn't see my eyes.** Face the camera with your face
evenly lit and nothing bright behind you, then say "Peek, calibrate" or choose
Calibrate Eyes. If the pill says "Camera problem", allow Camera access and
restart Peek.

**The outline lands next to what I'm looking at.** Say "Peek, re-center" and
look at the dot. If it's off in different ways in different parts of the
screen, recalibrate instead.

**Peek can't do anything on screen, or says it can't find things.** Allow
Accessibility for the app you launched Peek from, then restart Peek.

**Peek cuts itself off or answers its own voice.** Use headphones, or set
`PEEK_BARGE_IN=false`.

**Peek reacts to things I said to someone else.** After each exchange Peek
keeps listening for `PEEK_CONVERSATION_S` seconds (15 by default). Say
"thanks" to end the conversation right away, or lower the setting.

**Things get clicked when I didn't mean to.** That's wink-to-click. Set
`PEEK_WINK_CLICK=false`.

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
