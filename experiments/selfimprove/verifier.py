"""The reward interface and its bootstrap implementation.

Both training loops grade a trajectory with one function:

    reward(trajectory) -> float

trajectory here is a characterisation the caller chooses to score — the
harness transcript, its summary, or a check the agent produced. The interface
is deliberately dumb: a number in, a number that says "how good was that run".

`fake` exists so the whole online loop can be exercised end-to-end *before* the
DeepSeek judge is wired in — it never looks at the content, which is exactly
what makes it a good bootstrap: the plumbing is proven, then the judge is
swapped in and the loop must still run.
"""

import random


def fake(trajectory, rng=None):
    """Score a rollout without looking at it. Mind-useful:
      - any trajectory that 'did something' (non-empty) beats an empty one,
      - and real-looking scores come out of the same reward shape the judge
        will produce, so downstream code (advantages, tables, LoRA) is
        exercised for real.

    Returns 0.0 for an empty trajectory, else a float in [0, 1].
    """
    if not trajectory:
        return 0.0
    rng = rng or random
    return round(rng.random(), 3)


# The real judge will be dropped in here by Stage 6's app setup; callers
# resolve through this handle so nothing above the interface cares.
REWARD = fake