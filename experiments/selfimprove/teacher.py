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

try:
    import back.provider as _provider
    _DEEPSEEK = _provider.complete  # captured before any local-provider patch
except Exception:                   # authoring box: no `src` on sys.path
    _provider = None
    _DEEPSEEK = None

from . import traj

MODEL = _provider.DEFAULT_MODEL if _provider else "deepseek-chat"

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


def grade(messages):
    """Judge one trajectory -> float in [0, 10]. Falls back to 0 when the
    judge is unavailable (no DeepSeek key / network), so callers can count on
    always getting a number.
    """
    if _DEEPSEEK is None:
        return 0.0
    reply = _DEEPSEEK(MODEL, JUDGE_SYSTEM,
                      [{"role": "user",
                        "text": f"USER REQUEST:\n{messages[0].get('text','')}"
                                f"\n\nTRANSCRIPT:\n{_transcript_text(messages)}"}],
                      [])
    score = parse_grade(reply["text"])
    return score if score is not None else 0.0


def extract_golden(issue, messages):
    """Distil the golden reasoning trace of a well-scored run -> dict with the
    trace text. Returns None when the teacher is unavailable or the extract
    came back empty (caller then skips banking).
    """
    if _DEEPSEEK is None:
        return None
    request = (f"USER REQUEST:\n{issue}\n\n"
               f"TRANSCRIPT:\n{_transcript_text(messages)}")
    reply = _DEEPSEEK(MODEL, GOLDEN_SYSTEM,
                      [{"role": "user", "text": request}], [])
    trace = parse_golden(reply["text"])
    if not trace:
        return None
    return {"issue": issue, "golden": trace}