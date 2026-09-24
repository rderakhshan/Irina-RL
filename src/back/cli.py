"""Day 5 — the command line: the harness as something a person can talk to.

Everything underneath is a library, and a library has no opinion about
terminals. This file is where the two meet: it turns flags into a `Harness`,
events into lines you can read while they happen, and a permission question
into a prompt that waits for a human.

Two modes, one object. `-p` runs a task headlessly and exits, which is the
shape a script or a CI job wants; without it you get a prompt loop, which is
the shape a person wants. The only real difference between them is who is
watching — so headless defaults to yolo and interactive defaults to safe.

Design rules this file embodies:
  - Display is the caller's job, not the loop's. Everything here hangs off
    `on_event`, which is why no other module knows what a terminal is.
  - Show the call, not the intention. The approval prompt prints the arguments
    that will actually run, because that is the thing being approved.
  - Ctrl-C stops the run, not the session. The transcript is already on disk;
    the message says so, and says how to pick it up again.
"""

import argparse
import os
import sys

from .harness import Harness
from .security import Policy

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
ARG_CHARS = 100    # of a tool's arguments, on the call line
TEXT_CHARS = 4000  # of one assistant message, before we stop printing


def main(argv=None):
    """Parse the command line and run the harness, headless or interactive."""
    args = _parse(argv if argv is not None else sys.argv[1:])
    # A person can answer a prompt; a pipe cannot. So the default mode follows
    # who is on the other end, and an explicit --mode always wins.
    mode = args.mode or ("yolo" if args.prompt else "safe")
    policy = Policy(mode, approver=_approve)
    agent = Harness(workdir=args.workdir, model=args.model, policy=policy,
                    on_event=_print_event, max_turns=args.max_turns)

    if args.resume and agent.resume():
        print(f"{DIM}resumed {os.path.basename(agent.session_path)} "
              f"— {len(agent.messages)} messages{RESET}")

    if args.prompt:
        print(agent.run(args.prompt))
        return 0
    return _interactive(agent, mode)


def _parse(argv):
    """Build the argument parser. The flags are the whole public surface."""
    parser = argparse.ArgumentParser(
        prog="odysseus", description="A small coding agent that works in one "
                                     "directory.")
    parser.add_argument("-p", "--prompt",
                        help="Run one task headlessly and exit")
    parser.add_argument("-d", "--workdir", default=".",
                        help="Directory the agent may touch (default: .)")
    parser.add_argument("-m", "--model",
                        help="Model to use (default: $ODYSSEUS_MODEL)")
    parser.add_argument("--mode", choices=["safe", "yolo", "read-only"],
                        help="Permission mode (default: safe, yolo with -p)")
    parser.add_argument("--resume", action="store_true",
                        help="Continue the most recent session here")
    parser.add_argument("--max-turns", type=int, default=120,
                        help="Turns before the agent must wrap up")
    return parser.parse_args(argv)


def _print_event(kind, payload):
    """Render one event. Prose plainly, calls as one line, results dimmed.

    A tool result can be a whole file, and a terminal that prints it has hidden
    the story in the noise. One dimmed line under the call is what a reader
    needs — the model still gets every character.
    """
    if kind == "assistant":
        if payload["text"]:
            print(f"\n{_clip(payload['text'], TEXT_CHARS, flat=False)}")
        for call in payload["tool_calls"]:
            print(f"{BOLD}› {call['name']}{RESET} "
                  f"{_clip(call['args'], ARG_CHARS)}")
    elif kind == "tool_end":
        head = str(payload["result"]).strip().splitlines() or [""]
        more = f" … +{len(head) - 1} lines" if len(head) > 1 else ""
        print(f"{DIM}  {_clip(head[0], ARG_CHARS)}{more}{RESET}")


def _clip(value, limit, flat=True):
    """Shorten a value for the terminal, saying how much was left out."""
    text = str(value)
    if flat:
        text = text.replace("\n", " ")
    return text if len(text) <= limit else f"{text[:limit]}… [{len(text)} chars]"


def _approve(call, reason):
    """Ask the human. Anything that is not a clear yes is a no.

    The call is printed in full above the question: an approval given without
    seeing the arguments is not an approval, it is a habit.
    """
    print(f"\n{BOLD}› {call['name']}{RESET} {_clip(call.get('args'), 400)}")
    try:
        answer = input(f"  approve {call['name']}? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        # Refusing to answer is refusing. Never fall through to a yes.
        print()
        return False
    return answer in ("y", "yes")


def _interactive(agent, mode):
    """Run the prompt loop until end-of-file, and return an exit code."""
    print(f"{BOLD}Odysseus{RESET} — {agent.model} · {mode} · {agent.workdir}")
    print(f"{DIM}Type a task. Ctrl-C interrupts a run, Ctrl-D exits.{RESET}")
    while True:
        try:
            task = input(f"\n{BOLD}» {RESET}").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            # At the prompt there is no run to stop, so this is just a blank
            # line — quitting is Ctrl-D, and only Ctrl-D.
            print()
            continue
        if not task:
            continue
        try:
            print(f"\n{agent.run(task)}")
        except KeyboardInterrupt:
            # The transcript was written as the run happened, so there is
            # nothing to save here — only something to tell the user.
            print(f"\n{DIM}interrupted. The session log is safe: "
                  f"{agent.session_path or '(none yet)'}\n"
                  f"Continue it later with --resume.{RESET}")
        except RuntimeError as error:
            # A missing key or a rejected request should end the turn, not the
            # session: the user can fix the environment and try again.
            print(f"{DIM}error: {error}{RESET}")


if __name__ == "__main__":  # pragma: no cover - `python3 -m odysseus` is the way
    raise SystemExit(main())
