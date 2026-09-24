"""Day 3 demo — the same agent, told who it is and what this project knows.

Two things change here and neither is code the agent runs. The system prompt is
now assembled by `memory.build_system_prompt`, so anything written to
ODYSSEUS.md is present at the start of every future run; and the skills
catalogue is appended to it, so the agent can pull in a document it was not
started with.

The demo to run twice: `--remember "some fact"` in one process, then a question
about that fact in a completely fresh one. Nothing is shared between the two
runs except a Markdown file on disk, which is the whole idea.

Run it with:
  python3 demos/day3_memory.py --remember "The deploy command is ./ship.sh"
  python3 demos/day3_memory.py "What is the deploy command?"
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from odysseus import loop, memory, provider, security, skills, tools  # noqa: E402


def use_skill_tool(workdir):
    """Build the tool the skills catalogue tells the model to call.

    Loading is a tool call rather than an automatic injection because the model
    is the only part of the system that knows whether a skill applies to the
    task in front of it.
    """
    @tools.tool("Load the full instructions for a named skill and follow them",
                name="The skill name, exactly as listed in the catalogue")
    def use_skill(name):
        """Return the skill's whole SKILL.md for the model to read and obey."""
        return skills.read_skill(workdir, name)

    return use_skill


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


def main():
    """Either write one fact to project memory, or run a task that can read it."""
    workdir = os.environ.get("ODYSSEUS_WORKDIR") or tempfile.mkdtemp(prefix="odysseus-")
    args = sys.argv[1:]
    if args and args[0] == "--remember":
        # No model involved: memory is a file, and writing to it is a file write.
        print(memory.remember(workdir, " ".join(args[1:])))
        return

    task = args[0] if args else "Say hello and stop."
    system = memory.build_system_prompt(workdir, extra=skills.catalog_prompt(workdir))
    print(f"[workdir] {workdir}")
    print(f"[skills] {', '.join(skills.catalog(workdir)) or 'none'}")
    print(f"[system] {len(system)} chars, "
          f"memory {'loaded' if memory.MEMORY_FILE in system else 'absent'}")
    print(f"[user] {task}")

    kit = {t.name: t for t in tools.core_tools(workdir)}
    kit["use_skill"] = use_skill_tool(workdir)

    answer = loop.run_loop(
        model=provider.DEFAULT_MODEL,
        system=system,
        messages=[{"role": "user", "text": task}],
        tools=kit,
        on_event=on_event,
        before_tool=security.Policy("yolo").check,
    )
    print(f"\n[final] {answer}")


if __name__ == "__main__":
    main()
