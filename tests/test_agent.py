import json
from types import SimpleNamespace

from peek.agent import agent as agent_mod
from peek.agent.tools import Answer, Executor
from peek.screen.context import Registry, Snapshot


def block(kind, **fields):
    return SimpleNamespace(type=kind, **fields)


def tool_use(name, input, id="t1", text=None):
    content = [block("text", text=text)] if text else []
    content.append(block("tool_use", id=id, name=name, input=input))
    return SimpleNamespace(stop_reason="tool_use", content=content)


def final(text):
    return SimpleNamespace(stop_reason="end_turn", content=[block("text", text=text)])


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def make_agent(monkeypatch, responses):
    fake = FakeMessages(responses)
    monkeypatch.setattr(
        agent_mod.anthropic, "Anthropic", lambda api_key: SimpleNamespace(messages=fake)
    )
    return agent_mod.Agent("key", "model"), fake


def snapshot():
    return Snapshot({"gaze": {"x": 10, "y": 20}}, Registry(), image_png=b"png")


def handle(agent, said, **kwargs):
    return agent.handle(said, snapshot(), lambda: snapshot(), lambda quick: None, **kwargs)


def stub_tools(monkeypatch, result="ok"):
    ran = []
    monkeypatch.setattr(
        agent_mod.Executor,
        "run",
        lambda self, name, args: (ran.append((name, args)) or result, None),
    )
    return ran


def test_runs_tools_then_returns_spoken_reply(monkeypatch):
    ran = stub_tools(monkeypatch, "entered full screen")
    agent, fake = make_agent(
        monkeypatch,
        [tool_use("window_action", {"window_id": "w1", "action": "fullscreen"}), final("Done.")],
    )
    reply = handle(agent, "make this full screen")

    assert reply == "Done."
    assert ran == [("window_action", {"window_id": "w1", "action": "fullscreen"})]
    first = fake.calls[0]["messages"][-1]["content"]
    assert [c["type"] for c in first] == ["text", "image", "text"]
    tool_result = fake.calls[1]["messages"][-1]["content"][0]
    assert tool_result == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "entered full screen",
    }


def test_remembers_what_was_said_and_done(monkeypatch):
    stub_tools(monkeypatch, "closed")
    agent, fake = make_agent(
        monkeypatch,
        [
            tool_use("window_action", {"window_id": "w2", "action": "close"}),
            final("Closed Safari."),
            final("Opened it again."),
        ],
    )
    handle(agent, "close it")
    handle(agent, "undo that", said_name=False)

    content = fake.calls[2]["messages"][0]["content"]
    history = content[0]["text"]
    assert '"close it"' in history
    assert "window_action" in history and "closed" in history
    assert '"Closed Safari."' in history
    assert "didn't say your name" in content[-1]["text"]


def test_speaks_text_that_comes_with_tool_calls(monkeypatch):
    stub_tools(monkeypatch)
    spoken = []
    agent, _ = make_agent(
        monkeypatch,
        [tool_use("open_app", {"name": "Mail"}, text="Let me open Mail."), final("")],
    )
    reply = handle(agent, "check my mail", say=spoken.append)

    assert spoken == ["Let me open Mail."]
    assert reply == ""


def test_cancelling_stops_before_the_next_step_and_is_remembered(monkeypatch):
    ran = []
    reason = None

    def cancelled():
        return reason

    def run(self, name, args):
        nonlocal reason
        ran.append(name)
        reason = "They told you to stop."
        return "ok", None

    monkeypatch.setattr(agent_mod.Executor, "run", run)
    agent, fake = make_agent(
        monkeypatch,
        [
            tool_use("scroll", {"direction": "down"}),
            final("never reached"),
            final("Okay."),
        ],
    )
    assert handle(agent, "keep scrolling", cancelled=cancelled) is None
    assert ran == ["scroll"]

    reason = None
    handle(agent, "what's this")
    history = fake.calls[1]["messages"][0]["content"][0]["text"]
    assert "They told you to stop." in history


def test_silent_when_not_addressed(monkeypatch):
    agent, _ = make_agent(monkeypatch, [final(agent_mod.SILENT)])
    assert handle(agent, "did you see the game", said_name=False) == agent_mod.SILENT


def test_ask_returns_answer_and_what_they_looked_at(monkeypatch):
    monkeypatch.setattr(
        "peek.agent.tools.describe",
        lambda element, registry, x, y: {"id": registry.add_element(element), "role": "AXButton"},
    )
    executor = Executor(
        snapshot(),
        lambda: snapshot(),
        lambda quick: None,
        ask=lambda question: Answer("that one", element=object(), point=(5, 5)),
    )
    result, _ = executor.run("ask", {"question": "Which one?"})

    out = json.loads(result)
    assert out["answer"] == "that one"
    assert out["looking_at_while_answering"]["id"] == "e1"
    assert executor.registry.elements["e1"] is not None


def test_ask_without_answer():
    executor = Executor(snapshot(), lambda: snapshot(), lambda quick: None)
    result, _ = executor.run("ask", {"question": "Which one?"})
    assert "no answer" in result
