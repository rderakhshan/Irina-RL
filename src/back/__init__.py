"""Odysseus — a coding agent harness in ten files and no dependencies.

    from odysseus import Harness
    print(Harness("./project").run("build me a landing page"))

The package exports five names. `Harness` is the agent; `Policy` chooses how
much it may do without asking; `Tool` and `tool` are how you give it a hand it
did not ship with; `run_fleet` runs many of them at once. Everything else is an
internal module, importable when you want to read it — which is the point of
keeping it small.
"""

from .fleet import run_fleet
from .harness import Harness
from .security import Policy
from .tools import Tool, tool

__all__ = ["Harness", "Policy", "Tool", "tool", "run_fleet"]
