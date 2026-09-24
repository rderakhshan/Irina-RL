"""Day 2 demo — the same loop as yesterday, now with hands.

Nothing in `loop.py` changed. The only difference between the dice agent and an
agent that writes and runs code is what got passed into `tools` and
`before_tool` — which is the entire argument for keeping policy out of the loop.

Watch for the moment the agent writes a file, runs it, reads the output and
only then answers. That check is not in the loop and not in the tools; it comes
from the system prompt asking for it, and it is the difference between an agent
that claims and an agent that knows.

Run it with:  python3 demos/day2_build.py ["your task here"]
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from odysseus import loop, provider, security, tools  # noqa: E402

SYSTEM = """You are a coding agent working inside one directory.

Use your tools rather than describing what you would do. When you write code,
run it and read the output before you answer — never claim a result you have
not seen. Say plainly when a tool refuses you; do not try to work around it.
Keep the final answer to a couple of sentences."""

TASK = ("Create fib.py with an iterative fib(n), a __main__ printing fib(30), "
        "run it and confirm the output is 832040")


def on_event(kind, payload):
    """Print the transcript as it happens: the whole observability story."""
    if kind == "assistant":
        if payload["text"]:
            print(f"\n[assistant] {payload['text']}")
        for call in payload["tool_calls"]:
            print(f"[tool call ] {call['name']}({_short(call['args'])})")
    elif kind == "tool_end":
        print(f"[tool result] {_short(payload['result'])}")


def _short(value, limit=300):
    """Clip a value for display. The model sees all of it; the terminal need not."""
    text = str(value).replace("\n", "\\n")
    return text if len(text) <= limit else f"{text[:limit]}... [{len(text)} chars]"


def main():
    """Run one task in a scratch directory with a yolo policy."""
    task = sys.argv[1] if len(sys.argv) > 1 else TASK
    workdir = os.environ.get("ODYSSEUS_WORKDIR") or tempfile.mkdtemp(prefix="odysseus-")
    print(f"[workdir] {workdir}")
    print(f"[user] {task}")

    # A list of tools becomes the name-keyed dict the loop indexes calls by.
    kit = {t.name: t for t in tools.core_tools(workdir)}
    # yolo: everything is allowed except the handful of unrecoverable commands
    # in DENY_PATTERNS, which no mode can turn off.
    policy = security.Policy("yolo")

    answer = loop.run_loop(
        model=provider.DEFAULT_MODEL,
        system=SYSTEM,
        messages=[{"role": "user", "text": task}],
        tools=kit,
        on_event=on_event,
        before_tool=policy.check,
    )
    print(f"\n[final] {answer}")


if __name__ == "__main__":
    main()
