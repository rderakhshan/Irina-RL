# Irina-RL Self-Improvement Plan

Teach the Odyssey harness (a ten-file, zero-dependency coding-agent library in
`src/back/`) to get better at its own job. Two training loops, both built
around the harness exactly as it ships — nothing in `src/back/` changes except
one temperature parameter.

```
                    ┌──────────── user                            ┌────────────┐
                    │                                             │  Drive: .env │
   chat (Gradio) ───┤                                             │  + insight   │
                    └─▶ DeepSeek (read-only judge on REAL loop)   │  pool JSON   │
                                 ▲                                └────────────┘
                                 │ grades                          ▲            ▲
   Qwen2.5-0.5B (the policy) ────┤                                  │ SFT jobs   │ GRPO groups
   = the harness' acting model   │                                  │            │
                                 │ (swap via provider_local.py)     │            │
           Harness ──  real tools + real transcript ──┐              │            │
                runs tasks, survives kill, owns memory  │            │            │
                                                        ▼            │            │
                              offline: golden trajectories ────────▶─┘            │
                              online:  group sampled rollouts ────────▶──────────┘
```

**Acting model (student/policy):** `Qwen/Qwen2.5-Coder-0.5B-Instruct` — small
enough to fine-tune on a Colab T4, strong enough to actually call tools.

**Judge/teacher (reward + extraction):** DeepSeek — the `deepseek-chat` model
the harness already speaks to via `provider.py`. It grades rollouts and, for
the offline loop, extracts the "golden" reasoning trace from a successful run.

**The invariant:** the harness never knows it is being trained. It keeps
talking the neutral message format (`user / assistant / tool`) to
`provider.complete(...)`. `provider_local.py` presents the same signature, so
a one-line monkey-patch swaps DeepSeek for Qwen under a running harness.

## The reward interface

Both loops grade a characterisation of a rollout. One signature everywhere:

```python
def reward(trajectory) -> float
```

| Hook | What it returns | Who uses it |
| --- | --- | --- |
| `verifier.fake` | scripted score (bootstrap only) | online loop, sanity run |
| `teacher.grade` | DeepSeek judge score on the real loop | online loop, production |
| `teacher.extract_golden` | golden reasoning trace (not a patch) | offline loop |

`teacher.py` never receives the task's expected answer, because the tasks our
harness performs are open-ended (build, fix, explain) — there is no hidden
patch to leak. The teacher judges the *journey*: did the agent use tools
sensibly, converge in reasonable turns, produce a working, self-contained
result?

## File map

| File | What it is | Exists |
| --- | --- | --- |
| `src/back/provider.py` | DeepSeek door; temperature becomes a parameter | ✔ |
| `experiments/selfimprove/provider_local.py` | Qwen acting provider, same signature | ✘ |
| `experiments/selfimprove/traj.py` | neutral transcript → trainable chat form | ✘ |
| `experiments/selfimprove/verifier.py` | fake reward for bootstrapping | ✘ |
| `experiments/selfimprove/grpo.py` | GRPO math lifted from SWE notebook | ✘ |
| `experiments/selfimprove/teacher.py` | DeepSeek judge + golden extractor | ✘ |
| `experiments/selfimprove/insight_pool.py` | pending pool of goldens on Drive | ✘ |
| `experiments/selfimprove/offline.py` | SFT on pooled goldens | ✘ |
| `experiments/selfimprove/tasks.py` | tiny task catalog for the online loop | ✘ |
| `experiments/selfimprove/colab_gradio.py` | the Gradio app | ✘ |
| `colab_selfimprove.ipynb` | the Colab notebook, section by section | ✘ |

## The Gradio app (the UI)

One screen, three lives, a status bar.

- **Chat tab (default).** Type a task; a real `Harness` with the Qwen policy
  runs it — reads files, runs commands, edits, remembers, compacts. The
  transcript is the raw material for learning.
- **Offline Learning.** A button, always pressable, showing pool readiness
  (`12 / 30`). Every time a chat session closes, `teacher.extract_golden`
  appends one candidate to the insight pool *automatically*. Pressing the
  button drains the pool into an SFT step on Qwen (marked below).
- **Online Learning.** A toggle. Turns on a loop that picks one task, samples
  a GRPO group on it, grades with DeepSeek, and does one update — then the
  *next* task, then the next, and the result becomes the new skill baseline;
  the same toggle turns it off between tasks.
- **Status bar.** Model name, LoRA active/absent, current Gradio-colab budget.

## Checkboxes

### Stage 0 — baseline ✔ (commit `f7490ef`)
- [x] Re-baseline the repo onto `origin/main` (repo had only a stub README).
- [x] `gitignore` `.env` (holds `DEEPSEEK_API_KEY`) and `.venv/` before
      anything was committed.
- [x] First commit carries the full working harness; pushed to GitHub.

### Stage 1 — plan + repo skeleton
- [x] This `PLAN.md` with the checkbox map.
- [ ] `experiments/__init__.py` and `experiments/selfimprove/__init__.py`
      (empty package markers + docstrings).
- [ ] Commit `chore: seed selfimprove package` and push.

### Stage 2 — the acting provider
- [ ] `provider.py`: `complete(..., temperature=0.4)` (one line, default
      unchanged — DeepSeek chat keeps its current behaviour).
- [ ] `provider_local.py`: load Qwen2.5-Coder-0.5B once, cache it, present
      `complete(model, system, messages, tools)` returning the same
      `{"text", "tool_calls", "usage"}` dict.
  - [ ] Neutral → Qwen chat template (function calling baked in).
  - [ ] Tool-call parse: `<tool_call>` blocks → `{"name", "args", "signature"}`.
  - [ ] Generation uses temperature 1.0 (exploration; see GRPO notebook) and,
        when asked, returns forced log-probs of the assistant tokens.
  - [ ] `install_provider()` monkey-patch helper: `provider.complete =
        provider_local.complete` — one line, swap in, swap out.
- [ ] Smoke: `traj`/`grpo` can re-read a Qwen transcript offline (no GPU
      needed on this machine for the conversion unit test).
- [ ] Commit `feat: qwen acting provider with temperature` and push.

### Stage 3 — GRPO engine
- [ ] `traj.py`: neutral transcript → chat-form messages that are *append-only*
      under `apply_chat_template` (the masking invariant).
  - [ ] Unit-testable against a recorded transcript from `experiments/smoke/`.
- [ ] `grpo.py`: `add_lora`, `build_masked`, `seq_logprob`, `grpo_step` —
      lifted from `swe_grpo_one_step.py` (SWE-bench GRPO teaching demon),
      minus pandas display, plus a plain-text table.
  - [ ] `(r - mean)/std` advantages, REINFORCE `-(adv*logp)/G`, AdamW, grad
        clip, LoRA attached only when training starts.
- [ ] `verifier.py`: `fake(trajectory) -> float` so the online loop can be
      proven end-to-end before the DeepSeek judge is wired.
- [ ] Commit `feat: grpo engine from swe notebook` and push.

### Stage 4 — the teacher and the pool
- [ ] `teacher.py`: DeepSeek judge.
  - [ ] `grade(trajectory) -> float` — prompt lists the tool transcript;
        returns a score with a one-line rationale. No task answer involved.
  - [ ] `extract_golden(issue, trajectory) -> dict` — pulls the reasoning
        trace out of a run the judge scored well; this is the offline SFT
        target, never a hidden patch.
- [ ] `insight_pool.py`: pending pool, persisted as JSON on Drive.
  - [ ] `append`, `status()` (count + approx tokens), `threshold` (default 30
        goldens), `drain()` (atomic take-all), session-close hook.
  - [ ] Checks `teacher.grade` before accepting a golden, so the pool holds
        *good* runs only.
- [ ] Commit `feat: deepseek teacher and insight pool` and push.

### Stage 5 — offline SFT + online task catalog
- [ ] `offline.py`: `sft_step(model, tok, goldens)` — teacher-forced
      cross-entropy over assistant tokens only (same masking as GRPO), LoRA
      on, loss printed before/after, no reward — plain imitation of goldens.
- [ ] `tasks.py`: 3–4 self-contained coding tasks (one broken Python file +
      a hidden assert) the online loop can be pointed at repeatedly.
- [ ] Commit `feat: offline sft and online task catalog` and push.

### Stage 6 — the Gradio UI
- [ ] `colab_gradio.py`: chat tab + offline button (with pool readiness) +
      online toggle + status bar, per the app spec above.
- [ ] Threading: online loop runs in a background thread, toggle sets a stop
      flag read between tasks (never mid-episode).
- [ ] Does not edit `src/back/*` beyond the temperature line.
- [ ] Commit `feat: gradio ui for chat, offline, online` and push.

### Stage 7 — the Colab notebook
- [ ] `colab_selfimprove.ipynb`, one section per idea, each with a markdown
      cell explaining **logic · goal · need** before the code runs:
      1. Runtime + install (torch, transformers, peft, datasets, gradio,
         accelerate; `pip uninstall -y torchao` — peft floor).
      2. Mount Drive, clone/pull the repo, `sys.path`, load `.env`.
      3. The harness unchanged — what it is, why we never edit it.
      4. The acting provider (Qwen) and the one-line swap.
      5. The transcript → training-example converter.
      6. The reward (fake → DeepSeek judge).
      7. One GRPO step on a real transcript (proves the math).
      8. The insight pool (offline live).
      9. Offline SFT: drain pool, loss goes down, a re-run is visibly better.
      10. Online loop: toggle on, watch it fix task N, toggle off.
      11. The Gradio app: launch `demo.launch(share=True)`.
- [ ] Runs top-to-bottom on a fresh T4 with no manual edits.
- [ ] Commit `docs: colab notebook with per-section logic, goal, need` and push.

### Stage 8 — final pass
- [ ] `README.md`: add an `Irina-RL` section linking `PLAN.md`, the notebook,
      and the two loops.
- [ ] `python -m py_compile` every `.py` file added in stages 2–6.
- [ ] Full `git log` tells the story; push final commit; report to the user.

## Acceptance criteria

1. **Offline loop moves the model:** before/after SFT on the same golden, the
   loss demonstrably drops and a re-run of the same task is at least
   comparable (ideally better) under the DeepSeek judge.
2. **Online loop moves the model:** GRPO on one task raises the log-prob of
   the higher-rewarded rollout (the notebook's before/after table shows it),
   and the loop walks one task → next → next until the toggle is pressed.
3. **Nothing in `src/back/` changes except the temperature line.**
4. **Everything runs from the Colab notebook**, cell by cell, fresh runtime.
5. **Every stage is a commit with a self-explanatory title, pushed to
   `https://github.com/rderakhshan/Irina-RL`.**