"""Day 3 demo — a task deliberately too long for its context budget.

The budget here is 1500 tokens, which a handful of file writes blows straight
through. Watch the [compact] lines: the history collapses to a summary plus the
last few turns, and the agent carries on with the task anyway — it still knows
which files it has made, because the summary was told to keep exactly that.

The loop is again untouched. Compaction arrives through `before_turn`, the
socket day 1 left empty, and the loop never learns that the history it holds is
not the history it started with.

Run it with:  python3 demos/day3_compact.py ["your task here"]
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from odysseus import context, loop, provider, security, tools  # noqa: E402

BUDGET_TOKENS = 1500

SYSTEM = """You are a coding agent working inside one directory.

Use your tools rather than describing what you would do. Follow the requested
order of operations exactly, one tool call at a time. Never claim a result you
have not seen in a tool result. Keep the final answer to a couple of sentences."""

TASK = ("Create five files one.txt through five.txt, each with 20 lines of the "
        "word ping, one write_file at a time with a read back after each; then "
        "MANIFEST.md listing each file and its line count verified with wc -l")


def on_event(kind, payload):
    """Print the transcript as it happens: the whole observability story."""
    if kind == "assistant":
        if payload["text"]:
            print(f"\n[assistant] {payload['text']}")
        for call in payload["tool_calls"]:
            print(f"[tool call ] {call['name']}({_short(call['args'])})")
    elif kind == "tool_end":
        print(f"[tool result] {_short(payload['result'])}")


def _short(value, limit=200):
    """Clip a value for display. The model sees all of it; the terminal need not."""
    text = str(value).replace("\n", "\\n")
    return text if len(text) <= limit else f"{text[:limit]}... [{len(text)} chars]"


def before_turn(messages):
    """Compact if needed, and say so — the demo's only addition to `compact`.

    Announcing it here rather than inside `context.py` keeps that module free of
    display decisions, the same way the loop is free of policy.
    """
    before = context.estimate_tokens(messages)
    kept = context.compact(provider.DEFAULT_MODEL, messages, BUDGET_TOKENS)
    if len(kept) != len(messages):
        print(f"\n[compact] {len(messages)} messages / ~{before} tokens -> "
              f"{len(kept)} messages / ~{context.estimate_tokens(kept)} tokens")
    return kept


def main():
    """Run one long task under a budget small enough to force compaction."""
    task = sys.argv[1] if len(sys.argv) > 1 else TASK
    workdir = os.environ.get("ODYSSEUS_WORKDIR") or tempfile.mkdtemp(prefix="odysseus-")
    print(f"[workdir] {workdir}")
    print(f"[budget] {BUDGET_TOKENS} tokens, keeping the last {context.KEEP_RECENT} messages")
    print(f"[user] {task}")

    kit = {t.name: t for t in tools.core_tools(workdir)}
    policy = security.Policy("yolo")

    answer = loop.run_loop(
        model=provider.DEFAULT_MODEL,
        system=SYSTEM,
        messages=[{"role": "user", "text": task}],
        tools=kit,
        on_event=on_event,
        before_tool=policy.check,
        before_turn=before_turn,
    )
    print(f"\n[final] {answer}")


if __name__ == "__main__":
    main()
