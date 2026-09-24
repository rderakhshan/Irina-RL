"""GRPO training math, lifted from the SWE-bench teaching demo
`swe_grpo_one_step.py` and adapted to the harness' neutral message format.

Credits: the math (group-relative advantages, REINFORCE with a group baseline,
seq-logprob over masked assistant tokens, LoRA attached at training time) is
the demonstration from `abgoswam/swe_in_prod_vizuara_01`. That notebook
rewards a *patch*; here reward is `verifier.reward(trajectory)`, and the
trajectory is the QA transcript the harness produced — for our open-ended
tasks there is deliberately no hidden patch to leak.

The one hard requirement on segmentation: `traj.to_chat` must render
append-only (prefixes are real prefixes), which `grpo.build_masked` asserts as
it walks the transcript, exactly like the demo. That invariant is unit-tested
in `test_traj.py`.

imports of torch stay *inside* functions so this module can be imported and
type-checked on a laptop with no torch — only the training path needs it.
"""

from contextlib import nullcontext

import numpy as np

from . import traj

DEFAULT_LR = 1e-5
GRAD_CLIP = 1.0
MAX_SEQ = 3072


def _torch():
    import torch  # deferred: no GPU anywhere near the authoring box
    return torch


def add_lora(model, r=8, lora_alpha=16):
    """Attach trainable LoRA head. Called only when training starts — the
    acting model runs its rollouts in plain fp16/fp32, then gains the adapter
    for the update, mirroring the demo."""
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    return model


def build_masked(messages, system, tokenizer, max_len=MAX_SEQ):
    """Token ids + labels with everything but the assistant turns masked -100.

    `messages` is the *neutral* transcript; it is converted to chat form here
    so the masking segments are exactly the same blocks `provider_local.complete`
    generates (one canonical view of a trajectory). An assert keeps the demo's
    append-only promise: no template, no position in history, may silently
    re-render earlier turns.
    """
    chat = traj.to_chat(messages, system=system)
    ids, labels, prev = [], [], ""
    for i, m in enumerate(chat):
        cur = tokenizer.apply_chat_template(chat[:i + 1], tokenize=False)
        assert cur.startswith(prev), "chat template is not append-only"
        seg = tokenizer(cur[len(prev):], add_special_tokens=False)["input_ids"]
        ids += seg
        labels += seg if m["role"] == "assistant" else [-100] * len(seg)
        prev = cur
    return ids[:max_len], labels[:max_len]


def seq_logprob(model, tok, messages, system, device):
    """Mean log-prob over supervised (assistant) tokens, plus how many."""
    torch = _torch()
    ids, labs = build_masked(messages, system, tok)
    t = torch.tensor([ids], device=device)
    msk = torch.tensor([[0. if l == -100 else 1. for l in labs]],
                       device=device)[:, 1:]
    logits = model(t).logits[:, :-1]
    lp = torch.log_softmax(logits.float(), -1).gather(
        -1, t[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (lp * msk).sum() / msk.sum().clamp(min=1), msk.sum().item()


def grpo_step(model, tok, group, rewards, system, device, lr=DEFAULT_LR):
    """One GRPO update: rewards -> group-relative advantages -> REINFORCE step.

    `group` is a list of neutral transcripts, all from rollouts of the SAME
    task; `rewards` aligns 1:1 with it. A greedy zero-mean group (all rewards
    equal) is the honest failure mode of a small model — reported, not
    scripted over. Returns {"loss", "grad_norm", "advantages", "logp_before",
    "logp_after"} for the notebook tables.
    """
    torch = _torch()
    rewards = torch.tensor(rewards, dtype=torch.float)
    adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr)
    logp_before = [seq_logprob(model, tok, g, system, device)[0].item()
                   for g in group]

    model.train()
    opt.zero_grad(set_to_none=True)
    loss_total = 0.0
    for g, a in zip(group, adv):
        lp, _ = seq_logprob(model, tok, g, system, device)
        loss = -(a.to(device) * lp) / len(group)
        loss.backward()
        loss_total += loss.item()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
    opt.step()

    logp_after = [seq_logprob(model, tok, g, system, device)[0].item() for g in group]

    return {"loss": loss_total,
            "grad_norm": float(grad_norm),
            "advantages": adv.numpy().round(3).tolist(),
            "logp_before": [round(b, 4) for b in logp_before],
            "logp_after": [round(c, 4) for c in logp_after],
            "rewards": rewards.numpy().round(3).tolist()}


def pretty(step_result):
    """Render the step result as a readable table for the notebook."""
    rows = [f"{'rollout':>7} {'reward':>8} {'adv':>7} {'logp→':>8} {'sup':>4}"]
    sup = step_result.get("sup_tokens") or [None] * len(step_result["rewards"])
    for i, r in enumerate(step_result["rewards"]):
        rows.append(f"{i:>7} {r:>8} {step_result['advantages'][i]:>7} "
                    f"{step_result['logp_after'][i]:>8} "
                    f"{'–' if sup[i] is None else sup[i]:>4}")
    return "\n".join(rows)


def degenerate(rewards):
    """True when a reward group carries no signal (all equal)."""
    return float(np.std(rewards)) < 1e-8