from types import SimpleNamespace

from deixis.agent import agent as agent_mod
from deixis.screen.context import Registry, Snapshot


def block(kind, **fields):
    return SimpleNamespace(type=kind, **fields)


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


def test_runs_tools_then_returns_spoken_reply(monkeypatch):
    ran = []
    monkeypatch.setattr(
        agent_mod.Executor,
        "run",
        lambda self, name, args: (ran.append((name, args)) or "entered full screen", None),
    )
    agent, fake = make_agent(
        monkeypatch,
        [
            SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    block(
                        "tool_use",
                        id="t1",
                        name="window_action",
                        input={"window_id": "w1", "action": "fullscreen"},
                    )
                ],
            ),
            SimpleNamespace(stop_reason="end_turn", content=[block("text", text="Done.")]),
        ],
    )
    reply = agent.handle("make this full screen", snapshot(), lambda: snapshot(), lambda: None)

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


def test_remembers_previous_turns(monkeypatch):
    agent, fake = make_agent(
        monkeypatch,
        [
            SimpleNamespace(stop_reason="end_turn", content=[block("text", text="Which one?")]),
            SimpleNamespace(stop_reason="end_turn", content=[block("text", text="Closed.")]),
        ],
    )
    agent.handle("close it", snapshot(), lambda: snapshot(), lambda: None)
    agent.handle("the left one", snapshot(), lambda: snapshot(), lambda: None)

    roles = [m["role"] for m in fake.calls[1]["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert fake.calls[1]["messages"][1]["content"] == "Which one?"
