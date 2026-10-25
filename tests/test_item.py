from peek.screen import ax


class Node:
    def __init__(self, role, *children, subrole=None, name=None):
        self.attrs = {"AXRole": role, "AXSubrole": subrole, "AXDescription": name}
        self.attrs["AXChildren"] = list(children)
        for child in children:
            child.attrs["AXParent"] = self

    def __repr__(self):
        return self.attrs["AXDescription"] or self.attrs["AXRole"]


def article(*children):
    return Node("AXGroup", *children, subrole="AXDocumentArticle")


def post(author, likes):
    text = Node("AXStaticText", name=f"{author} says hi")
    quoted = article(Node("AXLink", name="quoted author"), Node("AXButton", name="quoted like"))
    actions = Node(
        "AXGroup",
        Node("AXButton", Node("AXImage"), name="Reply"),
        Node("AXButton", Node("AXImage"), Node("AXStaticText"), name=f"{likes} Likes. Like"),
    )
    root = article(Node("AXGroup", Node("AXLink", name=author)), Node("AXGroup", text), quoted)
    root.attrs["AXChildren"].append(actions)
    actions.attrs["AXParent"] = root
    return root, text


def fake_ax(monkeypatch):
    monkeypatch.setattr(ax, "attr", lambda element, name: element.attrs.get(name))


def test_finds_the_post_and_only_its_own_controls(monkeypatch):
    fake_ax(monkeypatch)
    first, text = post("alice", 3)
    second, _ = post("bob", 9)
    Node("AXWebArea", Node("AXGroup", first, second))

    item = ax.item_of(text)
    assert item is first
    assert [repr(c) for c in ax.controls_in(item)] == ["alice", "Reply", "3 Likes. Like"]


def test_no_item_outside_articles_and_rows(monkeypatch):
    fake_ax(monkeypatch)
    text = Node("AXStaticText")
    Node("AXWebArea", Node("AXGroup", text, Node("AXButton")))
    assert ax.item_of(text) is None


def test_a_row_is_its_own_item(monkeypatch):
    fake_ax(monkeypatch)
    row = Node("AXRow", Node("AXCell", Node("AXButton", name="Flag")))
    assert ax.item_of(row) is row
    assert [repr(c) for c in ax.controls_in(row)] == ["Flag"]
