"""`python3 -m odysseus` — the package's front door.

It defers to `cli.main` so the entry point and the command line are separate
concerns: this file decides *that* the CLI runs, `cli.py` decides what it does.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
