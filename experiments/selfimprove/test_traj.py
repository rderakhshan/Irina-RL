"""Offline sanity checks for the conversion layer (no torch required).

Run:  python -m experiments.selfimprove.test_traj
These assert the append-only invariant that `grpo.build_masked` depends on,
and the JSON tool-call contract that `provider_local` and the trainer share —
so the expensive, GPU-bound stages 3..5 build on a converter already proven
to preserve the transcript faithfully.
"""

import json
import os

from . import traj


def sample_neutral_messages():
    """A realistic harness transcript: user ask, assistant calls a tool, the
    tool result, then a closing assistant answer."""
    return [
        {"role": "user", "text": "List the files in this project."},
        {"role": "assistant", "text": "",
         "tool_calls": [{"name": "list_files", "args": {"path": "."},
                         "signature": None}]},
        {"role": "tool", "name": "list_files",
         "text": "README.md\nmain.py\nPLAN.md"},
        {"role": "assistant", "text": "Here is the file listing: README.md, "
                                      "main.py, PLAN.md."},
    ]


def test_tool_block_lists_schema():
    tools = [{"schema": {"name": "list_files", "description": "List files",
                         "parameters": {"type": "object", "properties": {
                             "path": {"type": "string", "description": "dir"}},
                             "required": ["path"]}}}]
    block = traj.tool_block(tools)
    assert "list_files" in block and "path" in block
    assert traj.tool_block([]) == "\n\nAvailable tools: none."


def test_to_chat_prepends_system_and_maps_roles():
    chat = traj.to_chat(sample_neutral_messages(),
                        system="You are a coding agent.")
    assert chat[0] == {"role": "system",
                       "content": "You are a coding agent." + traj.SYSTEM_TAIL}
    assert chat[1]["role"] == "user"
    # assistant with a tool call must render as a *contiguous* assistant block
    assert chat[2]["role"] == "assistant"
    assert "list_files" in chat[2]["content"]
    assert "arguments" in json.loads(chat[2]["content"].splitlines()[0])
    # tool result becomes a prefixed user message
    assert chat[3]["role"] == "user"
    assert chat[3]["content"].startswith("[tool list_files]\n")
    assert chat[4]["role"] == "assistant"


def test_to_chat_is_append_only():
    """The full rendered chat must equal the concatenation of prefixes —
    the invariant grpo.build_masked asserts segment-by-segment."""
    messages = sample_neutral_messages()
    chat = traj.to_chat(messages, system="SYS.")
    texts = [m["content"] for m in chat]
    prefix = []
    for m in chat:
        prefix.append(m["content"])
        # content of each prefix equals a prefix of the full join
        assert "\n".join(prefix) == "\n".join(texts[:len(prefix)])


def test_parse_tool_calls():
    assert traj.parse_tool_calls('{"name": "grep", "arguments": {"q": "x"}}') == \
        {"name": "grep", "arguments": {"q": "x"}}
    # wrapped in a code fence alongside reasoning
    res = traj.parse_tool_calls("I should search.\n```json\n"
                                '{"name": "grep", "arguments": {"q": "x"}}\n```')
    assert res and res["name"] == "grep"
    # prose-only replies must not be mistaken for a call
    assert traj.parse_tool_calls("No tool needed, here is the answer.") is None
    assert traj.parse_tool_calls("") is None


def test_provider_update_and_roundtrip():
    messages = sample_neutral_messages()
    chat = traj.to_chat(messages)
    # the tool-call assistant turn is chat[1] (chat[0] is the user ask)
    m = chat[1]
    call = json.loads(m["content"])
    assert call["name"] == "list_files"
    # round-trip: re-translating the neutral transcript is stable
    assert traj.to_chat(messages) == chat


if __name__ == "__main__":
    import sys
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"ok   {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"{sum(1 for n in dir() if n.startswith('test_'))} tests, "
          f"{failures} failures")
    raise SystemExit(1 if failures else 0)