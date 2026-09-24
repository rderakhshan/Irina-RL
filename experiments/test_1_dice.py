"""Day 1 demo — the smallest possible agent: one hand-written tool, one loop.

The concept this demo teaches is that a "tool" is not a framework object. It is
anything with a `.spec` (the JSON schema the model reads) and a `.run` (the
Python the harness calls). Day 2 generates these; today we write one by hand so
the shape is unmistakable.

Design rules this file embodies:
  - The schema is the prompt. Its description is the only thing the model reads
    about the tool, so it is written for a reader, not for a validator.
  - Every argument is declared "string". Models emit JSON loosely, so the tool
    coerces at the boundary rather than trusting the type it asked for.

Run it with:  python3 demos/day1_dice.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from odysseus import loop, provider  # noqa: E402  (after the path bootstrap)


class RollDice:
    """A tool in its entirety: a schema the model reads and a function it calls."""

    spec = {"schema": {
        "name": "roll_dice",
        "description": "Roll count six-sided dice",
        "parameters": {
            "type": "object",
            "properties": {"count": {"type": "string", "description": "How many dice"}},
            "required": ["count"],
        },
    }}

    def run(self, count):
        """Roll `count` six-sided dice and return the rolls and their total."""
        # `count` arrives as a string because that is what the schema declared.
        n = int(str(count).strip())
        rolls = [random.randint(1, 6) for _ in range(n)]
        return f"rolls={rolls} total={sum(rolls)}"


def on_event(kind, payload):
    """Print the transcript as it happens: this is the whole observability story."""
    if kind == "assistant":
        if payload["text"]:
            print(f"\n[assistant] {payload['text']}")
        for call in payload["tool_calls"]:
            print(f"[tool call ] {call['name']}({call['args']})")
    elif kind == "tool_end":
        print(f"[tool result] {payload['result']}")


def allow_everything(call):
    """Day 1 permission policy: allow. Day 5 replaces this with a real one."""
    return None


def main():
    """Run the dice task and print the final answer plus token usage."""
    task = "Roll 3 dice and tell me whether the total beats 10"
    print(f"[user] {task}")
    messages = [{"role": "user", "text": task}]
    answer = loop.run_loop(
        model=provider.DEFAULT_MODEL,
        system="You are a concise assistant. Use the tools you are given.",
        messages=messages,
        tools={"roll_dice": RollDice()},
        on_event=on_event,
        before_tool=allow_everything,
    )
    print(f"\n[final] {answer}")


if __name__ == "__main__":
    main()
