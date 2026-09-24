"""Day 4 demo — a run that survives being killed, and one that delegates.

Two things arrive today. Every message is appended to a session file as it
happens, so `--resume` can pick up a run that was interrupted — including one
interrupted mid-tool-call, where `session.load` fills the unanswered calls with
a placeholder so the history is sendable again. And `spawn_agent` lets the
agent hand a self-contained task to a child with a clean context, getting back
only the child's final report.

`_Agent` below is a stand-in for tomorrow's `Harness`: the factory shape is what
`subagent_tool` needs, and tomorrow that class composes the whole week instead.

Run it with:
  python3 demos/day4_spine.py "your task"
  python3 demos/day4_spine.py --resume "carry on"
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from odysseus import (context, loop, memory, provider, security,  # noqa: E402
                      session, skills, subagent, tools)

BUDGET_TOKENS = 20000


def _short(value, limit=200):
    """Clip a value for display. The model sees all of it; the terminal need not."""
    text = str(value).replace("\n", "\\n")
    return text if len(text) <= limit else f"{text[:limit]}... [{len(text)} chars]"


class _Agent:
    """One agent: tools, policy, a session file, and the ability to make more.

    The parent and the child are the same class — a sub-agent is not a special
    kind of thing, it is another one of these with a deeper depth and its own
    empty message list.
    """

    def __init__(self, workdir, depth=0, path=None, messages=None):
        self.workdir = workdir
        self.depth = depth
        self.messages = messages if messages is not None else []
        # Only the parent keeps a session file. A sub-agent's context is meant
        # to be thrown away, so persisting it would be storing noise.
        self.path = path
        self.policy = security.Policy("yolo")
        self.tools = {t.name: t for t in tools.core_tools(workdir)}
        self.tools["spawn_agent"] = subagent.subagent_tool(self._child, depth)

    def _child(self, depth):
        """Factory handed to `subagent_tool`: a fresh agent, one level down."""
        return _Agent(self.workdir, depth=depth)

    def _record(self, message):
        """Append to the session file, if this agent has one."""
        if self.path:
            session.append(self.path, message)

    def on_event(self, kind, payload):
        """Print the transcript and persist it in the same breath."""
        pad = "  " * self.depth
        if kind == "assistant":
            self._record({"role": "assistant", "text": payload["text"],
                          "tool_calls": payload["tool_calls"]})
            if payload["text"]:
                print(f"\n{pad}[assistant] {payload['text']}")
            for call in payload["tool_calls"]:
                print(f"{pad}[tool call ] {call['name']}({_short(call['args'])})")
        elif kind == "tool_end":
            self._record({"role": "tool", "name": payload["call"]["name"],
                          "text": payload["result"]})
            print(f"{pad}[tool result] {_short(payload['result'])}")

    def run(self, task):
        """Run one task to completion and return the final text."""
        message = {"role": "user", "text": task}
        self.messages.append(message)
        self._record(message)
        return loop.run_loop(
            model=provider.DEFAULT_MODEL,
            system=memory.build_system_prompt(
                self.workdir, extra=skills.catalog_prompt(self.workdir)),
            messages=self.messages,
            tools=self.tools,
            on_event=self.on_event,
            before_tool=self.policy.check,
            before_turn=lambda msgs: context.compact(
                provider.DEFAULT_MODEL, msgs, BUDGET_TOKENS),
        )


def main():
    """Start a session, or resume the newest one and continue it."""
    args = sys.argv[1:]
    resume = args and args[0] == "--resume"
    task = args[1] if resume and len(args) > 1 else (args[0] if args and not resume
                                                     else "List the files here.")
    workdir = os.environ.get("ODYSSEUS_WORKDIR") or tempfile.mkdtemp(prefix="odysseus-")

    messages, path = [], None
    if resume:
        path = session.latest(workdir)
        if path:
            # The repair happens inside load: a history torn mid-tool-call comes
            # back with its holes filled, so it can be sent as-is.
            messages = session.load(path)
            print(f"[resume] {os.path.basename(path)} — {len(messages)} messages")
    if path is None:
        path = session.new_session(workdir, task)
        print(f"[session] {os.path.basename(path)}")

    print(f"[workdir] {workdir}")
    print(f"[user] {task}")
    agent = _Agent(workdir, path=path, messages=messages)
    print(f"\n[final] {agent.run(task)}")


if __name__ == "__main__":
    main()
