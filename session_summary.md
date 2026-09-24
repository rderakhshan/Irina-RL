# Session Summary: From the SWE GRPO Notebook to a Real Training Loop

A log of our working session: understanding SWE RL (GRPO) training via the
`abgoswam/swe_in_prod_vizuara_01` notebook, and investigating whether the
**Odysseus** harness can host that training loop in production.

---

## 1. What the notebook (swe_grpo_one_step.ipynb) is

A teaching demo that runs **one single GRPO training step** end-to-end on one
SWE-bench task, printing every intermediate number:

1. **The task** — one SWE-bench instance. The agent sees only a raw GitHub
   issue; the answer patch is hidden (visible to nobody), and hidden tests
   (`FAIL_TO_PASS` → must go red→green, `PASS_TO_PASS` → must stay green)
   define "fixed".
2. **The environment** — `MockEnv`, a Python dict holding one buggy file.
   State → action (bash) → new state + observation. The final diff *is* the
   candidate patch.
3. **The agent** — `agent = model + harness`. Model = `Qwen2.5-Coder-0.5B` (the
   policy, the only trainable part). Harness = deterministic loop:
   prepare → sample → parse bash block → execute → append, for 4 turns.
4. **The reward** — real one would inject `test_patch` and run the tests. The
   notebook uses a **leaky proxy**: 0 if no patch, a random score if any patch
   (it rewards "touching a file", not "fixing the bug" — a deliberate lesson).
5. **Sample a group** — run the agent 8× on the same task (7 real + 1
   scripted "oracle" rollout, so the group is never all-zeros).
6. **One GRPO step** — attach LoRA, mask everything except the model's own
   assistant tokens, compute **group-relative advantages**
   `A = (r − mean)/std`, then one REINFORCE step `−A·logp` with AdamW.
   The final table shows log-probs before/after: good trajectories nudge up,
   bad ones down.

**One-sentence version:** *One exam question → run the agent a handful of
times → grade each run → compare each run to its siblings → nudge the model so
better runs become more likely — repeat forever in production.*

---

## 2. Example: what happens if you give it a GitHub issue

- The agent reads the issue as its only input
- It explores via bash (`ls`, `cat`, `grep`) and sees printed output each turn
- It finds and edits the buggy file — the file diff is its solution patch
- Hidden tests grade the patch → reward or no learning signal

---

## 3. Notebook vs script (are they equivalent?)

**Same core pipeline, one functional difference.**

Identical, line-for-line: task filter, model load, harness loop, proxy reward,
LoRA config, masking, seq-logprob math, advantages, loss, grad-clip, AdamW,
seeds.

Differences:
- **Oracle absent in the script** — the notebook injects one perfect scripted
  rollout so the group has non-zero signal; the script samples 6 all-real
  rollouts, which are almost always all-zero rewards → **degenerate group →
  the update is a no-op**. The script even prints a "DEGENERATE GROUP" warning.
- Group size 8 (nb) vs 6 (script).
- Teaching scaffolding (field-by-field walkthrough, pretty printers) stripped.
- `PASS_TO_PASS` dropped (unused, since the proxy never runs tests).

**Verdict:** faithful "production-mode" version of the loop, minus the oracle.
Add the oracle and they're functionally equivalent.

---

## 4. Conceptual decomposition of the project

Confirmed with two refinements:

```
harness (4-turn agent loop)                      ← the trajectory producer
   └─ rolls out into the environment

training loop:
   1. run agent G times on same task (group)     ← the executor / code runner
   2. verifier scores each finished rollout → reward   ← reward estimator
   3. advantages = (reward − mean)/std (no critic)
   4. PEFT/LoRA gradient step on the LLM (−A·logp)     ← fine-tune policy
```

Refinements:
- The "code runner" is the **executor** (environment), not a judge — it runs
  commands and observes; the **verifier** judges only the final patch, after
  the episode ends.
- In this repo the verifier is **stubbed** (`reward_random`), not real tests.
- It's always the *same* policy being nudged (Qwen2.5-Coder-0.5B); LoRA adds
  tiny trainable adapters.

---

## 5. Can the Odysseus harness host this training loop?

Repo investigated: `D:\AI Engineering LAB\Odysseus` — "the smallest coding
agent harness that is still a real one": 10 files, zero dependencies, loop +
tools + security + context + memory + sessions + subagents + fleet.

### Maps well (rollout half, ready to use)

- **Real agentic loop** — `loop.run_loop()` (JSON function-calling, better than
  the notebook's bash-block parsing)
- **Complete trajectory captured** — `Harness.messages` in a neutral
  `user/assistant/tool` dict format — exactly what masked seq-logprob needs
- **Parallel group sampling** — `run_fleet` runs G harnesses concurrently
- **Environment/code runner** — path-jailed `bash`/`write_file`/`edit_file`
  tools in a real workdir
- **The seams exist** — one provider boundary, `extra_tools` hook,
  `before_tool` policy plug

### Missing (the layers you add *around* it)

- ❌ **No gradient path** — `provider.complete` is an HTTPS→DeepSeek call. GRPO
  needs local weights, per-token logprobs, `.backward()`, and a PEFT/LoRA
  update. You cannot backprop through an HTTP call.
- ❌ **No verifier/reward** — `run()` returns text; nothing scores it.
- ⚠️ Group semantics need wiring — `run_fleet` is isolation-first (different
  tasks/dirs), GRPO wants G rollouts of the *same* task.
- ⚠️ `temperature` hardcoded at 0.4 — too cold for GRPO sampling (notebook: 1.0).
- ⚠️ No per-token logprobs in provider replies.

### First verdict

> Half of it, by design: a great **rollout-generator** harness, but the
> training loop can't run on top of it as-shipped.

---

## 6. Docker — resolving the sandbox question

The harness *does* run real code by itself, but its "sandbox" is only a **soft
path jail** (`tools.py` `resolve` + `security.py` deny list), not isolation.
The README itself says: *"None of that makes an agent safe to point at a
directory you care about. Run it on a copy, or in a container."*

**Docker is the answer, and the codebase anticipates it** (`harness.py:99`:
*a caller can deliberately replace a core tool — a sandboxed `bash`, say*).

- **Option 1 (real sandbox):** run the whole harness inside one container per
  task — `docker run -v <workdir>:/project ...`. True isolation for all tools;
  the container also hosts the cloned repo + hidden tests.
- **Option 2 (partial):** keep the harness on host, swap only `bash` for a
  `docker exec` wrapper via `extra_tools`. Process isolated, file tools not.
- `yolo` permission mode pairs naturally with containers (no human approver
  needed when blast radius is the container).

---

## 7. Revised verdict (your question, updated)

**Yes — with the Docker layer added, Odysseus becomes a usable harness for the
SWE-GRPO training loop.** The sandbox blocker is now solved.

```
Docker container (repo @ base_commit + hidden tests)
   └─ Odysseus Harness  = the "code runner / env" + trajectory producer
        └─ local-policy provider (Qwen + logprobs)   ← swap-in at one seam
             └─ GRPO update (advantages + LoRA/PEFT) ← external generic code
```

- **Ready:** agent loop, full neutral-formatted trajectories, parallel group
  sampling, real sandbox via Docker.
- **To add (at the codebase's own seams):**
  1. local trainable policy that returns per-token logprobs (and expose
     temperature),
  2. a verifier (`test_patch` + `FAIL_TO_PASS`/`PASS_TO_PASS` runs → reward),
  3. the generic GRPO update code (advantages + LoRA gradient step).

---

## Next steps (natural follow-ups)

1. Prototype a `docker_harness` wrapper around Odysseus (one container per task).
2. Swap the provider seam for a local `transformers` model returning logprobs.
3. Add a `submit` tool + external verifier that runs hidden tests in-container.
4. Wire the generic GRPO step (advantages → LoRA/AdamW) as an outer loop over
   `run_fleet` rollouts.