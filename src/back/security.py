"""Day 2 — the policy: the one function that decides whether a tool may run.

Yesterday's loop called `before_tool` and did whatever it was told. That socket
is the only place permission belongs, and this file is what plugs into it. The
loop still knows nothing about safety; it just asks, and blocks on a string.

The model is not the adversary here — an ordinary model, reading a file that
tells it to run something, will try. So the check reads the *call*, never the
model's explanation of it, and the deny list wins over every mode including
yolo: a switch a user flips for convenience should not also arm a footgun.

Design rules this file embodies:
  - Deny by pattern, allow by category. Patterns catch the handful of commands
    that are unrecoverable; categories handle the ordinary rest.
  - A block returns a reason, not a raise. The model reads it, and usually
    proposes something narrower on the next turn.
  - Default to refusing. An approver that was never wired up says no.
"""

import re

# Reads cannot destroy anything, so they never need an approval prompt. Keep
# this set honest: a "read" tool that also writes belongs in the other half.
READ_TOOLS = {"read_file", "list_files", "grep"}

DENY_PATTERNS = [
    # Recursive force-delete aimed at the root, home, or $HOME. The lookaheads
    # accept any flag order (-rf, -fr, -r -f) and the tail accepts /, /*, ~/.
    r"\brm\b(?=[^;&|]*-[a-zA-Z]*r)(?=[^;&|]*-[a-zA-Z]*f)"
    r"[^;&|]*\s(/|~|\$HOME)[/*\s]*($|[;&|])",
    # Privilege escalation: a harness is the wrong layer to decide this.
    r"\bsudo\b",
    # Whole-device writes; neither has a small, recoverable mistake.
    r"\bmkfs\b",
    r"\bdd\s+if=",
    # Downloading a script and executing it unread, in one motion.
    r"\bcurl\b[^|]*\|\s*(sudo\s+)?\S*sh\b",
    # Rewriting shared history: the one git operation that destroys other
    # people's work rather than the caller's own.
    r"\bgit\s+push\b[^;&|]*(--force\b|\s-f\b)",
    # Redirecting onto a raw disk device rather than a file.
    r">\s*/dev/sd[a-z]",
]

MODES = ("read-only", "safe", "yolo")


class Policy:
    """The permission policy, shaped to be passed straight to `before_tool`.

    `mode` is one of "read-only" (reads only), "safe" (reads free, writes go to
    the approver) or "yolo" (everything except the deny list). `approver` is a
    callback `(call, reason) -> bool`; leaving it out means every write is
    refused, because a policy that silently allows is worse than a loud one.
    """

    def __init__(self, mode="safe", approver=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.approver = approver if approver is not None else _refuse

    def check(self, call):
        """Return None to allow the call, or a reason string to block it."""
        name = call.get("name", "")
        args = call.get("args") or {}

        # First and unconditionally: the commands no mode may run. This runs
        # before the yolo shortcut on purpose.
        if name == "bash":
            command = str(args.get("command", ""))
            for pattern in DENY_PATTERNS:
                if re.search(pattern, command):
                    return f"command matches a blocked pattern: {pattern}"

        if name in READ_TOOLS or self.mode == "yolo":
            return None
        if self.mode == "read-only":
            return f"{name} modifies state and the policy is read-only"

        # Safe mode: a human decides. Anything that is not a clear yes is a no,
        # including an approver that raises or returns something unexpected.
        if self.approver(call, f"{name} wants to run with {args}") is True:
            return None
        return f"{name} was not approved"


def _refuse(call, reason):
    """The default approver: refuse. Wiring one up is a deliberate act."""
    return False
