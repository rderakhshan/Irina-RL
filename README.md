# Odysseus

The smallest coding agent harness that is still a real one. Ten files, zero
dependencies, one directory of Python you can read in an afternoon.

It is not a toy. It writes files, runs commands, keeps a project memory,
compacts its own context when a task runs long, survives being killed mid-run,
delegates to sub-agents, and runs a fleet of itself in parallel. Everything a
coding agent does, at the size where you can still see how.

```python
from odysseus import Harness

print(Harness("./project").run("build me a landing page"))
```

## Running it

One environment variable, then one of three forms.

```bash
export ODYSSEUS_API_KEY=your-gemini-api-key   # ODYSSEUS_MODEL is optional
```

**A single task, headlessly.** Runs, prints the final answer, exits. This is
the shape for scripts and CI, so it defaults to `yolo` — nobody is watching to
approve anything.

```bash
python3 -m odysseus -p "add a --json flag to the CLI and test it" -d ./project
```

**A conversation.** A banner, then a prompt. Defaults to `safe`: reads are
free, and every write asks first. Ctrl-C stops a run, Ctrl-D leaves.

```bash
python3 -m odysseus -d ./project
```

**Continuing yesterday.** Picks up the most recent session in that directory,
including one that was killed in the middle of a tool call.

```bash
python3 -m odysseus --resume -d ./project
```

Other flags: `-m/--model` to override the model, `--mode {safe,yolo,read-only}`
to override the permission mode, `--max-turns` to change how long the agent may
work before it has to wrap up.

## The anatomy

Each file is one day of the workshop, and one idea. Read them in this order and
the harness assembles itself in front of you.

| Day | File | Lines | What it is |
| --- | --- | --- | --- |
| 1 | `provider.py` | 116 | One narrow doorway to the model. Swap vendors by rewriting one file. |
| 1 | `loop.py` | 72 | The agentic loop. Everything else is scaffolding around it. |
| 2 | `tools.py` | 220 | Six tools and a decorator to add your own. |
| 2 | `security.py` | 90 | Deny rules, permission modes, a path jail. |
| 3 | `context.py` | 90 | Budget and compaction — hours-long tasks in a finite window. |
| 3 | `memory.py` | 72 | `ODYSSEUS.md`: one convention is the whole memory system. |
| 3 | `skills.py` | 83 | Drop a `SKILL.md` in a folder; the agent gains expertise. |
| 4 | `session.py` | 102 | Append-only logs. Kill the process, resume mid-thought. |
| 4 | `subagent.py` | 47 | The harness inside the harness. |
| 5 | `harness.py` | 169 | Where the week composes. It owns no algorithm of its own. |
| 5 | `fleet.py` | 47 | Many harnesses, one goal. This built the projects in `demos/`. |
| 5 | `cli.py` | 149 | The front door: flags in, events out, a human in the loop. |

Line counts include the comments, which are half the point: this repository is
meant to be read, not vendored.

## Composing it

The harness is a library first. Give it a tool it did not ship with, and the
model can call it on the next turn — no registry, no plugin format, no base
class. The decorator derives the schema from the function signature, so the
description the model reads cannot drift from the code that runs.

```python
import json
import urllib.request

from odysseus import Harness, Policy, tool


@tool("Fetch a URL and return its body as text",
      url="The absolute URL to fetch")
def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8", "replace")[:20000]


agent = Harness(
    workdir="./project",
    policy=Policy("safe", approver=lambda call, why: input(f"{why} [y/N] ") == "y"),
    extra_tools=[fetch],
    system_extra="This project ships a JSON API; keep responses backwards compatible.",
)
print(agent.run("read the changelog at https://example.com/changelog and summarise it in NOTES.md"))
```

Two more seams worth knowing. A `skills/<name>/SKILL.md` file in the working
directory is offered to the agent by name and loaded only when it decides the
task needs it — a quality bar or a house style belongs there, not in the
prompt. And `run_fleet` runs many agents at once, one directory each:

```python
from odysseus import Harness, run_fleet

jobs = [{"name": "landing", "workdir": "./out/landing", "task": "build a landing page"},
        {"name": "cli", "workdir": "./out/cli", "task": "build a task-manager CLI"}]
for result in run_fleet(jobs, lambda workdir: Harness(workdir), max_workers=4):
    print(result["name"], "ok" if result["ok"] else "failed", result["report"][:200])
```

## Where the boundaries are

The path jail in `tools.py` keeps every tool inside the working directory, and
resolves symlinks before it checks, so a link pointing out is caught by where
it lands. The deny list in `security.py` outranks every permission mode,
including `yolo`, because a switch flipped for convenience should not also arm
a footgun. Sub-agents inherit the parent's jail and policy — delegation is not
a way around either — and never write a session file of their own, so
`--resume` can never wake up as somebody's child.

None of that makes an agent safe to point at a directory you care about. Run it
on a copy, or in a container, and read what it did.

---

# Irina-RL

A self-improvement experiment built on this harness: the same ten-file agent,
but its acting model is a small local `Qwen2.5-Coder-0.5B`, and a DeepSeek
teacher grades its transcripts so it can learn from its own work.

- **Online loop (GRPO):** sample a small group of rollouts of one task, score
  each transcript with the DeepSeek judge, take one group-relative-advantage
  REINFORCE step on the assistant-token log-probs.
- **Offline loop (SFT):** when a chat session closes, the teacher distills its
  best run into a "golden reasoning trace"; once enough goldens accumulate in
  the insight pool, one button drains the pool into an imitation step.
- **The invariant:** nothing in `src/back/` changes except a single temperature
  parameter. The harness keeps speaking its neutral `user / assistant / tool`
  message format; `provider_local.py` presents the same `complete(...)`
  signature, so a one-line monkey-patch swaps DeepSeek for Qwen under a running
  harness, and `teacher.py` binds the real DeepSeek door at import time so it
  can never grade with its own student.

Everything lives in `experiments/selfimprove/`, runs from the Colab notebook
`colab_selfimprove.ipynb` (section by section, fresh T4), and is driven by a
Gradio app with chat / offline / online tabs. The full roadmap and per-stage
checklist are in [PLAN.md](PLAN.md).
