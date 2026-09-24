"""Day 1 — the loop: the small piece of control flow that makes an agent.

An agent is a while-loop. Ask the model, run whatever tools it asked for, feed
the results back, ask again, stop when it answers in prose. Every later day of
this workshop adds scaffolding *around* this shape, never inside it.

Design rules this file embodies:
  - The loop never crashes because a tool did. A failure is an observation the
    model reads and reacts to, not a traceback the user has to read.
  - The loop holds no policy. Permission lives in `before_tool`, history in
    `before_turn`, display in `on_event`. This file only sequences.
"""

from . import provider


def run_loop(model, system, messages, tools, on_event, before_tool,
             max_turns=80, before_turn=None):
    """Run the agent until the model answers without calling a tool.

    `tools` maps a name to a Tool exposing `.spec` (a {"schema": ...} dict) and
    `.run` (a callable taking keyword arguments). `on_event(kind, payload)`
    fires with "assistant" after each reply and "tool_start" / "tool_end"
    around each execution. `before_tool(call)` returns None to allow or a
    reason string to block. `before_turn(messages)`, when given, returns the
    list to use for this turn. Returns the model's final text.
    """
    for _ in range(max_turns):
        if before_turn:
            # Day 3 plugs compaction in here. The result replaces the history
            # in place so the caller keeps holding the same list object.
            messages[:] = before_turn(messages)

        # Rebuilt each turn: later days let the agent gain tools mid-run.
        specs = [t.spec for t in tools.values()]
        reply = provider.complete(model, system, messages, specs)
        messages.append({"role": "assistant", "text": reply["text"],
                         "tool_calls": reply["tool_calls"]})
        on_event("assistant", reply)
        if not reply["tool_calls"]:
            return reply["text"]

        # Calls run in the model's order: it may be reading a file it just wrote.
        for call in reply["tool_calls"]:
            on_event("tool_start", call)
            result = str(_execute(call, tools, before_tool))
            messages.append({"role": "tool", "name": call["name"], "text": result})
            on_event("tool_end", {"call": call, "result": result})

    # Out of turns. One last call with no tools, so the model has no choice but
    # to summarise what it managed to do rather than start something new.
    messages.append({"role": "user", "text": "Turn limit reached; wrap up now."})
    reply = provider.complete(model, system, messages, [])
    messages.append({"role": "assistant", "text": reply["text"], "tool_calls": []})
    on_event("assistant", reply)
    return reply["text"]


def _execute(call, tools, before_tool):
    """Run one tool call, turning every possible failure into readable text."""
    reason = before_tool(call)
    if reason:
        return f"BLOCKED: {reason}"
    tool = tools.get(call["name"])
    if tool is None:
        return f"ERROR: unknown tool {call['name']}"
    try:
        return tool.run(**call["args"])
    except Exception as e:
        # Deliberately broad: a tool raising is data for the model, which
        # usually corrects its own arguments on the next turn.
        return f"ERROR: {type(e).__name__}: {e}"
