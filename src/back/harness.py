"""Day 5 — the Harness: the file where the week meets itself.

Every module so far solved one problem and knew nothing about the others. The
loop does not know what a policy is; the policy has never heard of a session;
compaction has no opinion about skills. That independence is what made each of
them small — and it leaves exactly one job undone, which is introducing them.

This class is that introduction and nothing else. It owns no algorithm: read it
looking for cleverness and you will not find any, because every decision was
already made in the file that owns it. A harness is composition made explicit.

    Harness(workdir).run("build me a website")

Design rules this file embodies:
  - Assemble, never re-decide. If a rule belongs to another module, call it;
    a second copy of a rule here is a bug waiting for the two to disagree.
  - A sub-agent is this same class. The child is built by the parent's own
    constructor, which is why delegation cost twenty-seven lines on day 4.
  - Children are ephemeral. A child never writes a session file, because
    `--resume` picks the newest one and must never wake up as a sub-agent.
  - Persist as it happens, not at the end. A transcript written on completion
    is a transcript that does not exist in the case it was built for.
"""

import os

from . import context, loop, memory, provider, session, skills
from .security import Policy
from .subagent import subagent_tool
from .tools import core_tools, tool


class Harness:
    """One agent, fully assembled: hands, a policy, a memory and a spine.

    The arguments are the seams of the week. `policy` is day 2, `budget_tokens`
    day 3, `session_path` and `enable_subagents` day 4 — each one a place where
    a caller can swap in their own behaviour without editing this file.
    """

    def __init__(self, workdir=".", model=None, policy=None, extra_tools=None,
                 system_extra="", on_event=None, budget_tokens=600_000,
                 max_turns=120, session_path=None, enable_subagents=True,
                 persist=True, _depth=0):
        self.workdir = os.path.realpath(workdir)
        # Created here rather than on first write: every module below assumes
        # the directory exists, and one mkdir is cheaper than six guards.
        os.makedirs(self.workdir, exist_ok=True)
        self.model = model or os.environ.get("ODYSSEUS_MODEL",
                                             provider.DEFAULT_MODEL)
        # Yolo by default because the library caller is a program, not a person
        # at a terminal; the CLI is what asks a human and passes "safe" in.
        self.policy = policy or Policy("yolo")
        self.on_event = on_event or (lambda kind, payload: None)
        self.budget_tokens = budget_tokens
        self.max_turns = max_turns
        self.session_path = session_path
        self.persist = persist
        self.messages = []

        # Name-keyed because that is what the loop looks a call up by. The
        # dict is built in layers, later layers free to shadow earlier ones.
        self.tools = {t.name: t for t in core_tools(self.workdir)}

        @tool("Save a durable note to project memory (ODYSSEUS.md); it is "
              "loaded into every future session in this directory.",
              note="The fact worth remembering")
        def remember(note):
            return memory.remember(self.workdir, note)
        self.tools["remember"] = remember

        # Only offered when the project has skills. A tool the model cannot
        # use successfully is a tool it will waste a turn discovering.
        skill_prompt = skills.catalog_prompt(self.workdir)
        if skill_prompt:
            @tool("Load a skill's full instructions into context.",
                  name="Name of the skill to load")
            def use_skill(name):
                return skills.read_skill(self.workdir, name)
            self.tools["use_skill"] = use_skill

        if enable_subagents:
            def make_child(depth):
                """The factory day 4 asked for: this class, one level down.

                The child inherits the sandbox and the policy — it must not be
                a way around either — and inherits neither the conversation nor
                the session file, which is the whole point of delegating.
                """
                return Harness(workdir=self.workdir, model=self.model,
                               policy=self.policy, on_event=self.on_event,
                               budget_tokens=budget_tokens,
                               max_turns=max_turns, enable_subagents=True,
                               persist=False, _depth=depth)
            self.tools["spawn_agent"] = subagent_tool(make_child, depth=_depth)

        # Merged last so a caller can deliberately replace a core tool — a
        # sandboxed `bash`, say — rather than being told which names are ours.
        for extra in extra_tools or []:
            self.tools[extra.name] = extra

        # Built once, not per turn: the system prompt is what the agent *is*,
        # and a prompt that changes underneath a run is a prompt nobody can
        # debug. Skills come before the caller's text so a caller can override.
        self.system = memory.build_system_prompt(
            self.workdir,
            "\n\n".join(part for part in (skill_prompt, system_extra) if part))

    def resume(self, path=None):
        """Adopt a previous session's messages. True when there were any.

        With no path, the newest session in this directory — which is what a
        person means by "carry on", and why a sub-agent must never write one.
        """
        path = path or session.latest(self.workdir)
        if not path or not os.path.exists(path):
            return False
        # `load` repairs as it reads: a history torn mid-tool-call comes back
        # with truthful placeholders, so it is sendable without further care.
        self.messages = session.load(path)
        self.session_path = path
        return bool(self.messages)

    def _record(self, message):
        """Append one message to the session file, when there is one."""
        if self.session_path:
            session.append(self.session_path, message)

    def run(self, task):
        """Run one task to completion and return the model's final text."""
        if self.persist and self.session_path is None:
            # The label is only a filename: the first few words of the task are
            # what lets a human pick their run out of a directory listing.
            self.session_path = session.new_session(self.workdir, task[:32])

        user = {"role": "user", "text": task}
        self.messages.append(user)
        self._record(user)

        # Everything from here on is persisted by watermark: whatever the loop
        # has appended past `recorded` is new, whatever it is. The alternative
        # — recording from inside each event branch — has to be taught the
        # shape of every message kind, and forgets the one added next year.
        recorded = len(self.messages)

        def on_event(kind, payload):
            nonlocal recorded
            # Compaction replaces the history with a shorter list, so the
            # watermark can point past the end. Clamping first means the worst
            # case is re-recording a few messages, never skipping one — and the
            # file stays the true story even when memory no longer is.
            recorded = min(recorded, len(self.messages))
            for message in self.messages[recorded:]:
                self._record(message)
            recorded = len(self.messages)
            self.on_event(kind, payload)

        result = loop.run_loop(
            model=self.model, system=self.system, messages=self.messages,
            tools=self.tools, on_event=on_event,
            before_tool=self.policy.check, max_turns=self.max_turns,
            before_turn=lambda messages: context.compact(
                self.model, messages, self.budget_tokens))

        # The turn-limit path in the loop appends after the last event fires,
        # so one final flush is what keeps the transcript honest to the end.
        for message in self.messages[recorded:]:
            self._record(message)
        return result
