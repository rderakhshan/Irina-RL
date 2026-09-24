"""Day 4 — sub-agents: delegation as a tool, and the reason it is one.

Compaction (day 3) saves a run that has already filled its context. A sub-agent
avoids filling it. "Find every place this function is called" costs forty tool
results to answer and one sentence to report, and the parent only needs the
sentence — so the search happens in a context that is thrown away afterwards.

Nothing here is a new mechanism. A sub-agent is another harness, and the parent
reaches it through the same tool interface it uses to read a file. That is why
this module is thirty lines: recursion is free once the shape is right.

Design rules this file embodies:
  - The child is isolated on purpose. It cannot see the parent's conversation,
    which is what makes the delegation cheap — and what makes a vague task fail,
    so the description tells the model to send a self-contained one.
  - Depth is bounded. Any agent that can spawn agents can spawn them forever,
    and the limit is a plain counter passed down rather than global state.
  - The tool never raises. Hitting the limit is advice to do it yourself.
"""

from .tools import tool


def subagent_tool(make_harness, depth=0, max_depth=2):
    """Build the `spawn_agent` tool for a harness sitting at `depth`.

    `make_harness(depth)` returns a fresh harness — the parent hands over a
    factory rather than a child, so no sub-agent is built unless one is asked
    for, and each call gets a genuinely new context.
    """
    @tool("Delegate a self-contained task to a fresh sub-agent with its own "
          "clean context. The sub-agent cannot see this conversation, so the "
          "task must contain everything it needs. Returns the sub-agent's "
          "final report.",
          task="The complete, self-contained task for the sub-agent")
    def spawn_agent(task):
        """Run one task in a child harness and return only its final answer."""
        if depth >= max_depth:
            # Returned as text, not raised: the parent reads this and does the
            # work itself, which is the behaviour we actually want.
            return "ERROR: sub-agent depth limit reached; do this task yourself"
        child = make_harness(depth + 1)
        # Only the final report crosses back. The child's tool results stay in
        # the child's context and die with it — the whole point of delegating.
        return child.run(task)

    return spawn_agent
