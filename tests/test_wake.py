import pytest

from peek.wake import split_command

WAKE = ["peek", "hey computer"]


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("Peek, close this window.", "close this window."),
        ("Peek's make this full screen", "make this full screen"),
        ("Peak type hello world", "type hello world"),
        ("hey Peek, what's this?", "what's this?"),
        ("Hey computer split these two", "split these two"),
        ("Peek.", ""),
        ("Peek", ""),
        ("It's like so loud over here. Peek.", ""),
        ("navigate, wait, peek.", ""),
        ("what the heck? Hey Peek, can you create an API key?", "can you create an API key?"),
        ("Wait, wait, wait, wait. Hey Peek.", ""),
        ("Peek. You can scan it now. Yeah.", "You can scan it now. Yeah."),
    ],
)
def test_finds_command_after_wake_word(transcript, expected):
    assert split_command(transcript, WAKE) == expected


@pytest.mark.parametrize(
    "transcript",
    [
        "I was telling her about the meeting",
        "close this window",
        "let me take a peek at the window",
        "I said she took a peek. Then she left.",
        "",
    ],
)
def test_ignores_speech_without_leading_wake_word(transcript):
    assert split_command(transcript, WAKE) is None
