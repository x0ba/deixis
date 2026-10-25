import pytest

from peek.voice.echo import content_words, is_backchannel, strip_echo


@pytest.mark.parametrize(
    ("transcript", "spoken", "expected"),
    [
        ("Do you want me to close Safari?", ["Do you want me to close Safari?"], ""),
        (
            "Do you want me to close Safari? Yes, close it.",
            ["Do you want me to close Safari?"],
            "Yes, close it.",
        ),
        # Recognition garbles a couple of echoed words.
        (
            "Do you want me to clothes Safaris? No.",
            ["Do you want me to close Safari?"],
            "No.",
        ),
        # Interrupted partway: only the first part of the reply was heard.
        (
            "I opened the settings for you the blue one",
            ["I opened the settings for you, which one do you want?"],
            "the blue one",
        ),
        ("Done. Now open Mail.", ["Done."], "Now open Mail."),
        ("Wait, stop.", ["Here is the article. It starts with an introduction."], "Wait, stop."),
        ("open the window", ["Closed the window."], "open"),
        ("make it bigger", [], "make it bigger"),
    ],
)
def test_strips_own_voice(transcript, spoken, expected):
    assert strip_echo(transcript, spoken) == expected


def test_lone_common_word_is_kept():
    assert strip_echo("scroll to the top", ["I moved the window."]) == "scroll to the top"


def test_backchannel():
    assert is_backchannel("Okay.")
    assert is_backchannel("mm-hmm, yeah")
    assert not is_backchannel("okay close it")
    assert content_words("uh, close that window") == 3
