"""Day 3 — memory: the prompt the agent wakes up with, and the file it keeps.

There are two kinds of memory in a harness. Context is what the agent holds
during a run and loses at the end; this file is the other kind — a plain
Markdown file in the project that survives every run, gets read into the system
prompt on the next start, and can be edited by a human with an editor.

That last property is the point. A memory you cannot open and correct is not a
memory, it is a rumour. `ODYSSEUS.md` sits in the repository, gets committed,
reviewed and diffed like anything else the project knows about itself.

Design rules this file embodies:
  - The system prompt is assembled, not hard-coded. Base behaviour, environment
    and project memory are three separate concerns that meet here.
  - Behaviour lives in the prompt, not in Python. Every line of the base prompt
    exists to prevent a failure mode we have actually watched happen.
  - Missing memory is normal. A project with no ODYSSEUS.md yields a prompt
    without that section, not an error.
"""

import os
import sys

MEMORY_FILE = "ODYSSEUS.md"

BASE_PROMPT = """You are Odysseus, a small sharp coding agent. You work inside \
one directory, using the tools you have been given.

Act, don't narrate: call a tool rather than describing the call you would make.
Inspect before you assume — read the file, list the directory, run the command.
Prefer edit_file over write_file for small changes, so you do not throw away
work you did not read. After you build something, verify it: run it, or read it
back. Never repeat a failing call unchanged; change something or try another
way. When the task is complete, reply with a short summary and stop calling
tools."""


def build_system_prompt(workdir, extra=""):
    """Assemble the system prompt for a run in `workdir`.

    Sections are joined by blank lines because that is what a model reads as a
    boundary; the order is fixed — who you are, where you are, what this
    project knows, then whatever the caller adds (skills, on day 3).
    """
    root = os.path.realpath(workdir)
    parts = [BASE_PROMPT,
             f"You are running on {sys.platform}. "
             f"The working directory is {root}."]

    memory = os.path.join(root, MEMORY_FILE)
    if os.path.exists(memory):
        with open(memory, encoding="utf-8", errors="replace") as handle:
            # Read whole and inline: memory that has grown too big to fit is a
            # signal to prune the file, not to truncate it behind the user.
            parts.append(f"Project memory ({MEMORY_FILE}):\n{handle.read().strip()}")

    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


def remember(workdir, note):
    """Append one durable fact to the project's memory file.

    Append-only and one line per fact, so two writes can never lose each other
    and a human can delete a line they disagree with without a merge.
    """
    root = os.path.realpath(workdir)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, MEMORY_FILE), "a", encoding="utf-8") as handle:
        handle.write(f"- {note}\n")
    return f"Remembered in {MEMORY_FILE}"
