"""The Gradio control surface for the self-improvement loop.

This module is the UI for the whole experiment and nothing more: it owns no
training math and no judging logic, because those live in `grpo`, `offline`,
`teacher` and `insight_pool`. Its one job is to wire four harness-scale
actions to buttons and text boxes:

  - **Chat**    — run one interactive session (Qwen acting, temp ~0.4) and,
                 when it closes, bank its golden trace into the insight pool.
  - **Offline** — drain the pool and fine-tune on the collected goldens (the
                 button shows readiness and always works; it never blocks).
  - **Online**  — a background loop that rolls out one fixed task per
                 iteration, judges the group with DeepSeek, takes ONE GRPO
                 step, moves to the next task. The toggle stops *between*
                 tasks, never mid-episode.
  - **Status**  — a live line: model, LoRA attached?, pool count/threshold,
                 current loop state, last event.

Import ordering matters and is the one thing this file cannot do for you:
`teacher` must capture the real DeepSeek `complete` BEFORE
`provider_local.install_provider()` swaps in Qwen, so in the notebook import
this module first, then install the provider.

The two `run` paths below are the only place the harness is even mentioned in
the experiment — everything else talks to training math or the pool.
"""

import os
import threading

from . import grpo, insight_pool, offline, provider_local, teacher, traj

_HARNESS = None     # resolved on first use, so a late sys.path fix is honored


def _harness():
    """Resolve `back.harness.Harness` lazily. Resolving at import time cached
    a `None` forever when `src` wasn't on sys.path yet; resolving here means a
    late bootstrap fix (e.g. the notebook's section-2 self-bootstrap) works."""
    global _HARNESS
    if _HARNESS is None:
        from back.harness import Harness
        _HARNESS = Harness
    return _HARNESS


DEFAULT_GROUP_SIZE = 6       # GRPO rollouts per online task
DEFAULT_MAX_TURNS = 4        # short episodes: small window, no compaction
ROLLOUT_BUDGET = 4_000_000   # chars; far over budget so a rollout never compacts
CHAT_BUDGET = 800_000        # a real chat may legitimately compact

CHAT_SYSTEM_EXTRA = ("Answer in short prose. When the task is done, say what "
                     "you did and stop calling tools.")


def _system_prompt_for(workdir):
    """The harness builds system prompts per-workdir; the app mirrors that for
    the fixed chat workspace so SFT masks agree with what chat actually saw."""
    from back import memory
    return memory.build_system_prompt(workdir, extra=CHAT_SYSTEM_EXTRA)


def _acting_system(harness):
    """Exactly the system the acting provider rendered for generation:
    harness system prompt + rendered tool block. Masking must reuse this
    string or build_masked would segment against a *different* prefix than the
    one the model was generated under.
    """
    return harness.system + traj.tool_block(
        [t.spec for t in harness.tools.values()])


class SelfLearn:
    """A tiny controller so the Gradio callbacks stay one-liners.

    All mutable state lives here under one lock; the UI reads it through
    `status_text()`. The online loop runs in a daemon thread; keying on a
    threading.Event lets the UI stop it cleanly between tasks.
    """

    def __init__(self, workdir, pool_path=None, model_name=None,
                 group_size=DEFAULT_GROUP_SIZE, max_turns=DEFAULT_MAX_TURNS):
        self.workdir = os.path.realpath(workdir)
        os.makedirs(self.workdir, exist_ok=True)
        self.model_name = model_name or provider_local.DEFAULT_MODEL
        self.group_size = group_size
        self.max_turns = max_turns

        # The chat workspace is fixed so the system prompt (and thus any SFT
        # masking over chat traces) is stable across runs.
        self.chat_dir = os.path.join(self.workdir, "chat")
        os.makedirs(self.chat_dir, exist_ok=True)
        self._chat_system = _system_prompt_for(self.chat_dir)

        self.pool = insight_pool.InsightPool(
            pool_path or os.path.join(self.workdir, "insight_pool.json"),
            teacher=teacher)
        self.lora = None            # trained (PEFT) head, attached once
        self.device = None
        self.tokenizer = None

        self._lock = threading.Lock()
        self.log = []               # last N lines, newest last
        self.stop_online = threading.Event()
        self.online_thread = None
        self._task_index = 0
        self._last_chat = None

        try:
            _harness()
        except Exception:
            self._note("warning: back.harness not importable here — "
                       "this box has no `src` on sys.path (or no torch).")

    # -- shared plumbing ------------------------------------------------

    def _note(self, line):
        with self._lock:
            self.log.append(line)
            self.log = self.log[-40:]

    def _logtail(self, n=25):
        with self._lock:
            return "\n".join(self.log[-n:])

    def _trainable(self):
        """The acting model, with LoRA attached the first time it is needed.
        After attachment the acting provider is pointed at the *trained* head
        so later rollouts are generated by the improved student."""
        model, tok, device = provider_local.load(self.model_name)
        if self.lora is None:
            self.lora = grpo.add_lora(model)
            provider_local._LOADED["model"] = self.lora  # act as the learner
        self.tokenizer, self.device = tok, device
        return self.lora, tok, device

    def _run_harness(self, workdir, task_text, budget_tokens, temperature,
                     max_turns, enable_subagents=False, tag=""):
        """One rollout in `workdir`. Returns (messages, system, final_text)."""
        provider_local.TEMPERATURE = temperature
        h = _harness()(workdir=workdir, model=self.model_name,
                       system_extra=CHAT_SYSTEM_EXTRA,
                       budget_tokens=budget_tokens, max_turns=max_turns,
                       persist=False, enable_subagents=enable_subagents)
        final = h.run(task_text)
        self._note(f"[{tag}] done in {len(h.messages)} msgs")
        return h.messages, _acting_system(h), final

    # -- Chat tab ---------------------------------------------------------

    def chat(self, issue):
        """Run one live session with the student; on close, bank its golden."""
        try:
            messages, _, final = self._run_harness(
                self.chat_dir, issue, CHAT_BUDGET, 0.4, 120,
                enable_subagents=True, tag="chat")
            self._last_chat = (issue, final)
            bank = self.pool.bank(issue, messages)
            self._note(bank)
            return final, bank, self.status_text()
        except Exception as e:
            # Never a silent no-op: the UI should SEE the failure.
            self._note(f"[chat] ERROR: {e!r}")
            return (f"Error: {e}", f"chat failed — see `{e}`", self.status_text())

    # -- Offline tab --------------------------------------------------------

    def offline_status(self):
        st = self.pool.status()
        return (f"**{st['count']}/{st['threshold']} goldens** "
                f"({'ready' if st['ready'] else 'keep collecting…'}, "
                f"~{st['tokens']} words in pool)")

    def offline_apply(self):
        """Drain the pool and SFT the student on the collected goldens."""
        try:
            goldens = self.pool.drain()
            if not goldens:
                self._note("offline: pool empty — nothing to learn from")
                return "Pool is empty. Close a few chat sessions first.",
            model, tok, device = self._trainable()
            self._note(f"offline: sft on {len(goldens)} goldens")
            result = offline.sft_step(model, tok, goldens,
                                      system=self._chat_system, device=device)
            self._note(f"offline: loss {result['loss_before']}→"
                       f"{result['loss_after']} over {result['examples']} examples")
            return (f"Done: {result['examples']} goldens, cross-entropy "
                    f"{result['loss_before']} → {result['loss_after']}."), \
                   self.status_text()
        except Exception as e:
            self._note(f"[offline] ERROR: {e!r}")
            return f"Error: {e}", self.status_text()

    # -- Online tab ----------------------------------------------------------

    def online_toggle(self):
        """Flip the online loop; stop is requested between tasks, never mid-run."""
        if self.online_thread and self.online_thread.is_alive():
            self.stop_online.set()
            return "Stopping (waits for the current task to finish)…"
        self.stop_online.clear()
        self.online_thread = threading.Thread(target=self._online_loop,
                                              daemon=True)
        self.online_thread.start()
        return "Online learning started."

    def _online_loop(self):
        from . import tasks
        catalog = tasks.catalog()
        try:
            model, tok, device = self._trainable()
        except Exception as e:
            self._note(f"[online] cannot start: {e!r}")
            return
        while not self.stop_online.is_set():
            task = catalog[self._task_index % len(catalog)]
            self._task_index += 1
            root = tasks.scaffold_root()
            self._note(f"[online] task={task['id']} rolling out group of "
                       f"{self.group_size}")
            group, rewards = [], []
            for i in range(self.group_size):
                workdir = tasks.materialize(task, root)
                try:
                    messages, system, _ = self._run_harness(
                        workdir, task["task"], ROLLOUT_BUDGET, 1.0,
                        self.max_turns, enable_subagents=False,
                        tag=f"{task['id']}#{i}")
                except Exception as e:
                    self._note(f"[online] {task['id']}#{i} rollout failed: "
                               f"{e!r}")
                    continue
                group.append({"messages": messages, "system": system})
                rewards.append(teacher.grade(messages))

            if not group:
                self._note("[online] no rollouts this round — retrying")
                continue
            if grpo.degenerate(rewards):
                self._note(f"[online] {task['id']}: rewards {rewards} all "
                           "equal — no signal, step skipped")
                continue

            try:
                step = grpo.grpo_step(model, tok, group, rewards,
                                      system=None, device=device)
                self._note(f"[online] {task['id']}: "
                           f"rewards={step['rewards']} "
                           f"adv={step['advantages']} loss={step['loss']:.3f} "
                           f"grad={step['grad_norm']:.2f}")
            except Exception as e:
                self._note(f"[online] {task['id']} step failed: {e!r}")

    # -- Status -------------------------------------------------------------

    def status_text(self):
        st = self.pool.status()
        online = self.online_thread and self.online_thread.is_alive()
        model = getattr(provider_local, "_LOADED", {}).get("name") or self.model_name
        tb = teacher.binding()
        bound = "DeepSeek bound" if tb["bound"] else "**UNBOUND — grades will be 0**"
        lines = [
            f"**model:** {model}",
            f"**teacher:** {bound}",
            f"**pool:** {st['count']}/{st['threshold']} "
            f"({'ready' if st['ready'] else 'collecting…'})",
            f"**lora:** {'attached' if self.lora else 'not yet'}",
            f"**online:** {'running' if online else 'off'}",
            "---",
            self._logtail(),
        ]
        return "\n".join(lines)


def build_app(workdir=".", pool_path=None, **kwargs):
    """Assemble the Gradio Blocks. Imports gradio lazily so this module stays
    importable (and py_compile-able) on a machine without gradio."""
    import gradio as gr
    app = SelfLearn(workdir, pool_path=pool_path, **kwargs)

    with gr.Blocks(title="Irina-RL — Self-Improvement") as demo:
        status = gr.Markdown(app.status_text())

        with gr.Tab("Chat with the student"):
            issue = gr.Textbox(label="Ask the acting agent to do something",
                               lines=3, placeholder="Fix the bug in dice.py …")
            answer = gr.Markdown()
            bank_line = gr.Markdown()
            with gr.Row():
                send = gr.Button("Run session")
                clear = gr.Button("Clear")
            send.click(app.chat, inputs=issue,
                       outputs=[answer, bank_line, status])
            clear.click(lambda: ("", ""), outputs=[answer, bank_line])

        with gr.Tab("Offline learning"):
            pool_line = gr.Markdown()
            with gr.Row():
                pool_state = gr.Button("Show pool status")
                learn = gr.Button("Apply offline learning (SFT on pooled goldens)")
            out = gr.Markdown()
            pool_state.click(app.offline_status, outputs=pool_line)
            learn.click(app.offline_apply, outputs=[out, status])

        with gr.Tab("Online learning"):
            gr.Markdown("Rolls out one task at a time, judges the group, takes "
                        "one GRPO step, then moves on. The toggle stops "
                        "between tasks.")
            toggle = gr.Button("Start / stop online loop")
            toggle.click(app.online_toggle, outputs=status)
            refresh = gr.Button("Refresh status")
            refresh.click(app.status_text, outputs=status)

    return app, demo