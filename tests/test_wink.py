from peek.gaze.wink import WinkDetector

FPS = 30


def play(detector, frames, start=0.0):
    """Feed (left, right, seconds) segments; return the wink start times reported."""
    t, found = start, []
    for left, right, seconds in frames:
        for _ in range(round(seconds * FPS)):
            hit = detector.update(t, left, right)
            if hit is not None:
                found.append(hit)
            t += 1 / FPS
    return found


OPEN = (0.05, 0.05)


def test_wink_either_eye_is_reported_on_release():
    assert len(play(WinkDetector(), [(*OPEN, 0.5), (0.8, 0.1, 0.3), (*OPEN, 0.2)])) == 1
    assert len(play(WinkDetector(), [(*OPEN, 0.5), (0.1, 0.85, 0.3), (*OPEN, 0.2)])) == 1


def test_start_time_is_when_the_eye_shut():
    (start,) = play(WinkDetector(), [(*OPEN, 1.0), (0.8, 0.1, 0.3), (*OPEN, 0.2)])
    assert abs(start - 1.0) < 1.5 / FPS


def test_blinks_are_ignored():
    d = WinkDetector()
    assert play(d, [(*OPEN, 0.5), (0.9, 0.9, 0.15), (*OPEN, 0.5)]) == []
    # lopsided blink: one eye leads by a frame or two
    assert play(d, [(*OPEN, 0.5), (0.7, 0.2, 0.07), (0.9, 0.9, 0.1), (*OPEN, 0.5)], 2.0) == []


def test_too_short_or_too_long_is_ignored():
    assert play(WinkDetector(), [(*OPEN, 0.5), (0.8, 0.1, 0.1), (*OPEN, 0.5)]) == []
    assert play(WinkDetector(), [(*OPEN, 0.5), (0.8, 0.1, 2.0), (*OPEN, 0.5)]) == []


def test_squinting_both_eyes_is_not_a_wink():
    assert play(WinkDetector(), [(*OPEN, 0.5), (0.55, 0.4, 0.4), (*OPEN, 0.5)]) == []


def test_cooldown_between_winks():
    wink = [(0.8, 0.1, 0.25), (*OPEN, 0.1)]
    assert len(play(WinkDetector(), [(*OPEN, 0.5), *wink, *wink])) == 1
    assert len(play(WinkDetector(), [(*OPEN, 0.5), *wink, (*OPEN, 1.0), *wink])) == 2


def test_reset_drops_a_wink_in_progress():
    d = WinkDetector()
    play(d, [(*OPEN, 0.5), (0.8, 0.1, 0.3)])
    d.reset()
    assert play(d, [(*OPEN, 0.3)], 0.8) == []
