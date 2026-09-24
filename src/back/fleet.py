"""Day 5 — the fleet: many harnesses, one goal.

A sub-agent (day 4) is delegation inside one run: the parent waits for the
child's sentence and carries on thinking. A fleet is the other axis — several
independent runs, each in its own directory, working at the same time. Nobody
is anybody's parent, and no result crosses between them.

That isolation is what makes this file thirty lines rather than a scheduler.
Two agents editing one tree would need locks, merges and a conflict story; two
agents in two directories need a thread each. Give each job its own workdir and
concurrency stops being a design problem.

Design rules this file embodies:
  - One directory per job. It is the whole concurrency model.
  - A crashed job is a result, not an exception. One agent hitting a dead API
    key must not throw away the twenty-four reports that succeeded.
  - Results come back in input order. Threads finish in whatever order they
    like; a report the caller cannot line up with its job is a puzzle.
"""

from concurrent.futures import ThreadPoolExecutor


def run_fleet(jobs, make_harness, max_workers=4):
    """Run every job concurrently and return one result dict per job.

    `jobs` is a list of {"name", "workdir", "task"}; `make_harness(workdir)`
    builds the agent for one of them — a factory rather than a harness, because
    each job needs its own agent, its own session file and its own context.

    Returns [{"name", "ok", "report"}] in the order the jobs were given.
    """
    def run(job):
        try:
            return {"name": job["name"], "ok": True,
                    "report": make_harness(job["workdir"]).run(job["task"])}
        except Exception as error:
            # Deliberately broad, and the same reasoning as the loop's: the
            # failure is data the caller reads, not a traceback that kills the
            # other threads mid-build.
            return {"name": job["name"], "ok": False,
                    "report": f"{type(error).__name__}: {error}"}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # `map` is what keeps the ordering promise: it yields per input, not
        # per completion, so slow job one still comes back before fast job two.
        return list(pool.map(run, jobs))
