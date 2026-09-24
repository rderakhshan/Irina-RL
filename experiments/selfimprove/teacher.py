"""The teacher: a DeepSeek judge for the two training loops.

`grade` is the reward for GRPO — a number that says how good a trajectory was.
`extract_golden` distils a *reasoning trace* out of a good run; that trace is
the offline SFT target.

Two deliberate boundaries:

  - The teacher never sees a task's expected answer. Our tasks are open-ended
    (build, fix, explain) — there is no hidden patch, and judging the *journey*
    is what makes GRPO meaningful here.
  - `teacher` binds the DeepSeek `complete` ONCE, at import time. The acting
    provider (`provider_local`) monkey-patches `back.provider.complete` to
    point at Qwen; if the teacher looked the function up at call time it would
    end up judging with its own student. Capture early, judge with DeepSeek.

The parse helpers take real API strings and are unit-testable offline; only
`grade` / `extract_golden` touch the network.
"""

import sys

from . import traj

_provider = None
_DEEPSEEK = None
MODEL = "deepseek-chat"


def _capture():
    """(Re-)bind the DeepSeek `complete` from `src/back`, if it is on the
    path. Must run BEFORE `provider_local.install_provider()` swaps in Qwen,
    or the teacher would judge with its own student. Returns bound: bool."""
    global _provider, _DEEPSEEK, MODEL
    try:
        import back.provider as p
        _provider, _DEEPSEEK = p, p.complete
        MODEL = _provider.DEFAULT_MODEL if _provider else "deepseek-chat"
        return _DEEPSEEK is not None
    except Exception:           # authoring box / `src` not yet on sys.path
        _provider, _DEEPSEEK = None, None
        return False


_capture()


def rebind():
    """Re-capture DeepSeek after the bootstrap fixed `sys.path`. Call this from
    the notebook if section 5 reports the teacher unbound."""
    ok = _capture()
    if not ok:
        print("teacher.rebind(): `back` still not importable — check that "
              "REPO_DIR/src is on sys.path", file=sys.stderr)
    return ok


_warned = {"none": False, "error": False}


def binding():
    """Diagnostic summary for the UI / notebook."""
    return {"bound": _DEEPSEEK is not None, "model": MODEL,
            "via": (getattr(_provider, "__name__", "") if _provider else "")}

JUDGE_SYSTEM = """You are a rigorous, economical grader of a coding agent's
work. You read a transcript of an agent using tools (reading files, running
commands, editing) to answer a user request. Judge the JOURNEY: did the agent
explore the right files, use tools rather than guessing, attempt a sensible
fix, and finish with a clear, working-sounding answer? Score 0 (useless) to 10
(flawless). Reply with exactly one line: `SCORE <n> <one-line rationale>`."""

GOLDEN_SYSTEM = """You are a master coding agent distilling your own thinking
for a junior colleague. Given a user request and the transcript of an agent
that solved it well, produce the GOLDEN REASONING TRACE: a compact, stepwise
account an agent could imitate — what to read first, which clues matter, what
commands to run, what to edit, and finally a clear answer. 3 to 8 short steps,
plain text, no preamble."""


def _transcript_text(messages):
    """Render a neutral transcript as the readable text the judge reads."""
    chat = traj.to_chat(messages)
    lines = []
    for m in chat:
        role, content = m["role"], m["content"]
        if role == "system":
            continue
        lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines)


def parse_grade(text):
    """Pull the numeric score out of a judge reply. Finds `score`, then the
    first numeric token after it; robust to prose like "the score is 10",
    "Score: 8.5", or a bolded `SCORE 7`."""
    if not text:
        return None
    norm = text.replace("**", "").strip().lower()
    idx = norm.find("score")
    if idx < 0:
        return None
    tail = norm[idx + len("score"):]
    for token in tail.replace(",", " ").split():
        try:
            return float(token)
        except ValueError:
            continue
    return None


def parse_golden(text):
    """A golden trace is whoever the model finally 'answered with'. Clean it
    (fences, leading labels) but keep it verbatim otherwise — it IS the label."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned[cleaned.find("\n") + 1:].strip()
    return cleaned or None


def _bound_or_warn():
    """Ensure the teacher is bound; if not, warn once and return False."""
    if _DEEPSEEK is None:
        if not _warned["none"]:
            _warned["none"] = True
            print("teacher: NOT bound to DeepSeek (back.provider was not "
                  "importable at import time). Every grade returns 0 and "
                  "nothing banks. Run the notebook section 5 cell, then "
                  "teacher.rebind() before install_provider().",
                  file=sys.stderr)
        return False
    return True


def _guard_reply(reply, what, request_text):
    """Wrap the single external call so a DeepSeek outage is a loud visible
    message, not a silent 0.0 or a UI-window crash."""
    if _DEEPSEEK is None:
        _bound_or_warn()
        return None
    try:
        return _DEEPSEEK(MODEL, what,
                         [{"role": "user", "text": request_text}], [])
    except Exception as e:      # network/key error: surface, don't fake 0
        if not _warned["error"]:
            _warned["error"] = True
            print(f"teacher: DeepSeek call failed ({e!r}) — grades/extracts "
                  f"will be 0/none until it recovers", file=sys.stderr)
        return None


def _user_request(messages):
    """The boxed request the judge reads. Never touches the network."""
    boxed = ([m for m in messages if m.get("role") == "user"] or messages
             or [])[0]
    text = boxed.get("text", "") if boxed else ""
    return (f"USER REQUEST:\n{text}\n\n"
            f"TRANSCRIPT:\n{_transcript_text(messages)}")


def grade(messages):
    """Judge one trajectory -> float in [0, 10]. Returns 0 when the judge is
    unavailable (no DeepSeek key / network), with a loud stderr warning the
    first time, so callers can count on a number AND see why it's a 0."""
    if not _bound_or_warn():
        return 0.0
    reply = _guard_reply(messages, JUDGE_SYSTEM, _user_request(messages))
    if reply is None:
        return 0.0
    score = parse_grade(reply["text"])
    if score is None:
        print("teacher: judge reply carried no parseable `SCORE n` — "
              f"had nothing to grade on; reply: {reply['text'][:120]!r}",
              file=sys.stderr)
        return 0.0
    return score


def extract_golden(issue, messages):
    """Distil the golden reasoning trace of a well-scored run -> dict with the
    trace text. Returns None when the teacher is unavailable or the extract
    came back empty (caller then skips banking).
    """
    if not _bound_or_warn():
        return None
    request = (f"USER REQUEST:\n{issue}\n\n"
               f"TRANSCRIPT:\n{_transcript_text(messages)}")
    reply = _guard_reply(messages, GOLDEN_SYSTEM, request)
    if reply is None:
        return None
    trace = parse_golden(reply["text"])
    if not trace:
        return None
    return {"issue": issue, "golden": trace}