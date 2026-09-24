"""Day 3 — the context engine: what the harness does when the window fills.

A long task dies in a boring way. Twenty tool results in, the history no longer
fits, and the agent either errors or silently forgets the thing it was asked to
do. Compaction is the fix: replace the old middle of the conversation with a
summary the model wrote itself, and keep the recent turns verbatim.

The asymmetry is the whole idea. Recent messages are kept *exactly* because the
agent is mid-thought and needs the details; old messages are kept *in gist*
because what survives from them is the task, the files, and what went wrong.

Design rules this file embodies:
  - Compaction is lossy, so the summarising prompt names what may not be lost.
  - The summary re-enters as an ordinary message. Nothing downstream learns a
    new message kind, and the transcript stays one flat list.
  - Never orphan a tool result. A tool message that has lost the assistant turn
    that called for it is malformed history, and the provider will say so.
  - The loop is not told any of this happened; `before_turn` returns a list.
"""

from . import provider

CHARS_PER_TOKEN = 4      # good enough: we need a threshold, not an invoice
KEEP_RECENT = 6          # the tail kept verbatim — the agent's working memory
MAX_MESSAGE_CHARS = 1000  # per message, when rendering old turns for the summariser

SUMMARY_SYSTEM = ("You compress agent transcripts. Preserve: the original task, "
                  "every file created or edited and its purpose, key decisions, "
                  "unresolved errors, and what remains to be done. Be dense and "
                  "factual.")


def estimate_tokens(messages):
    """Estimate the history's size in tokens by its length in characters.

    Deliberately not a tokeniser: a tokeniser is a dependency, a version skew
    and a per-vendor rule, in exchange for precision this decision never needs.
    Four characters to the token is close enough to place a budget.
    """
    return sum(len(str(message)) for message in messages) // CHARS_PER_TOKEN


def compact(model, messages, budget_tokens):
    """Return the history to use this turn, summarising the old part if needed.

    Under budget, this is the identity function — the common case costs one
    length check and no API call. Over budget, everything before the last
    `KEEP_RECENT` messages becomes a single summary message.
    """
    # The second guard matters: with a short history there is nothing to gain
    # by summarising, and a summary call would cost more than it saves.
    if (estimate_tokens(messages) <= budget_tokens
            or len(messages) <= KEEP_RECENT + 1):
        return messages

    split = max(0, len(messages) - KEEP_RECENT)
    # Walk the split forward past any tool results that would start the kept
    # slice. They are answers to a call that is about to become summary text,
    # so they move into the summary with it rather than being dropped.
    while split < len(messages) and messages[split].get("role") == "tool":
        split += 1
    old, recent = messages[:split], messages[split:]

    reply = provider.complete(model, SUMMARY_SYSTEM,
                              [{"role": "user", "text": _render(old)}], [])
    summary = {"role": "user",
               "text": f"[Conversation so far, compacted]\n{reply['text']}"}
    return [summary] + recent


def _render(messages):
    """Flatten messages into a plain transcript for the summariser to read.

    Written for a reader, not a parser: the summariser is a model, so the
    cheapest legible format wins. Tool calls are named but their arguments are
    not — that a file was written matters, its bytes do not.
    """
    lines = []
    for message in messages:
        role = message.get("role", "?")
        # A tool result is only meaningful next to the tool that produced it.
        label = f"{role}({message['name']})" if message.get("name") else role
        text = str(message.get("text") or "")
        if len(text) > MAX_MESSAGE_CHARS:
            text = f"{text[:MAX_MESSAGE_CHARS]}... [{len(text)} chars]"
        called = ", ".join(call["name"] for call in message.get("tool_calls") or [])
        if called:
            text = f"{text} [calls: {called}]" if text else f"[calls: {called}]"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)
