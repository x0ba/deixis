import pytest

from deixis.wake import split_command

WAKE = ["deixis", "hey computer"]


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("Deixis, close this window.", "close this window."),
        ("Dixie's make this full screen", "make this full screen"),
        ("Daxis type hello world", "type hello world"),
        ("hey Deixis, what's this?", "what's this?"),
        ("Hey computer split these two", "split these two"),
        ("Deixis.", ""),
        ("Deixis", ""),
    ],
)
def test_finds_command_after_wake_word(transcript, expected):
    assert split_command(transcript, WAKE) == expected


@pytest.mark.parametrize(
    "transcript",
    [
        "I was telling her about the meeting",
        "close this window",
        "the word deixis comes from Greek",
        "",
    ],
)
def test_ignores_speech_without_leading_wake_word(transcript):
    assert split_command(transcript, WAKE) is None
