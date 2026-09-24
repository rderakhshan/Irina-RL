"""Day 4 — sessions: the transcript on disk, and what to do with a torn one.

A long agent run is a process that can be killed — by a laptop lid, an OOM, a
ctrl-C at the wrong moment. Everything the run knew lived in a list in memory,
so it is gone. Appending each message to a file as it happens turns that from a
lost afternoon into a resumable one.

The interesting part is not writing the file, it is reading a file that was
being written when the power went out. Two things are broken in it: the last
line may be half-written, and the last assistant turn may have asked for tools
that never ran. A provider rejects that second one outright — every tool call
must have a response — so `load` repairs it rather than handing back history
that cannot be sent.

Design rules this file embodies:
  - Append-only, one JSON object per line. A crash can damage the last line and
    nothing else, which is exactly the damage this format can survive.
  - Repair on read, not on write. The crashed process gets no chance to clean
    up after itself, so the next reader does it.
  - A repaired hole is visible. The placeholder says the call was interrupted,
    so the model can decide to run it again instead of trusting a blank.
"""

import json
import os
import re
import time

SESSION_DIR = ".odysseus/sessions"
INTERRUPTED = "Interrupted before this ran (process restarted)."


def new_session(workdir, label="session"):
    """Return the path for a fresh session file, creating the directory.

    The name is timestamp-first so the directory sorts chronologically in any
    listing, and slug-second so a human can tell the runs apart.
    """
    root = os.path.join(os.path.realpath(workdir), SESSION_DIR)
    os.makedirs(root, exist_ok=True)
    # Anything that is not alphanumeric or a dash becomes a dash: a label comes
    # from a task string and must not be able to name a path of its own.
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", label).strip("-").lower()[:40] or "session"
    return os.path.join(root, f"{int(time.time())}-{slug}.jsonl")


def append(path, message):
    """Append one message to the session file as a single JSON line."""
    with open(path, "a", encoding="utf-8") as handle:
        # ensure_ascii=False keeps non-English text readable in the file; the
        # newline is what makes a torn tail cost one message rather than all.
        handle.write(json.dumps(message, ensure_ascii=False) + "\n")


def load(path):
    """Read a session back, discarding a torn tail and repairing the history."""
    messages = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                # A half-written final line is the signature of a crash. There
                # is nothing valid after it, so stop rather than skip.
                break
    return repair(messages)


def latest(workdir):
    """Return the most recently written session file, or None."""
    root = os.path.join(os.path.realpath(workdir), SESSION_DIR)
    if not os.path.isdir(root):
        return None
    files = [os.path.join(root, name) for name in os.listdir(root)
             if name.endswith(".jsonl")]
    # By modification time, not by name: the session you want to resume is the
    # one that was still being written, not the one that was started last.
    return max(files, key=os.path.getmtime) if files else None


def repair(messages):
    """Give every unanswered tool call a result, in place, and return the list.

    A process killed between "the model asked for three tools" and "the third
    one finished" leaves history the provider will refuse. Filling the gap with
    a truthful placeholder is both what makes it sendable and what tells the
    resumed agent which work never happened.
    """
    last = None
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            last = index
    if last is None:
        return messages

    calls = messages[last].get("tool_calls") or []
    answered = sum(1 for message in messages[last + 1:]
                   if message.get("role") == "tool")
    for call in calls[answered:]:
        messages.append({"role": "tool", "name": call.get("name", ""),
                         "text": INTERRUPTED})
    return messages
