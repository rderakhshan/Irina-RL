"""Offline learning: SFT on the insight pool's golden reasoning traces.

GRPO is reward-driven learning. This is the other side: *imitation*. The pool
holds goldens distilled from sessions the DeepSeek judge scored well — each
one a `{"issue", "golden"}` pair. SFT turns each golden into a two-turn
training example:

    user:   <issue>
    assistant: <golden reasoning trace>

and runs teacher-forced cross-entropy over the assistant tokens only,
masks-turn-by-turn with the same `grpo.build_masked` the GRPO machine uses, so
the two learning paths share one faithfulness guarantee: nothing outside an
assistant turn is ever labelled.

`add_lora` is attached here before training, exactly like GRPO. The function
returns the loss before/after so the notebook can show the pool actually moved
the model.
"""

from . import grpo


def _golden_example(golden):
    """A pool entry -> the neutral two-turn transcript SFT expects."""
    return [
        {"role": "user", "text": golden.get("issue", "")},
        {"role": "assistant", "text": golden.get("golden", "")},
    ]


def _sft_loss(model, tok, examples, system, device):
    """Mean per-token cross-entropy over assistant turns, all examples pooled.
    The loss the training loop minimises; also used to report before/after."""
    torch = grpo._torch()
    total, count = 0.0, 0
    for example in examples:
        ids, labs = grpo.build_masked(example, system, tok)
        t = torch.tensor([ids], device=device)
        msk = torch.tensor([[0. if l == -100 else 1. for l in labs]],
                           device=device)[:, 1:]
        logits = model(t).logits[:, :-1]
        lp = torch.log_softmax(logits.float(), -1).gather(
            -1, t[:, 1:].unsqueeze(-1)).squeeze(-1)
        ce = -(lp * msk)
        total += ce.sum().item()
        count += msk.sum().item()
    return total / max(count, 1)


def sft_step(model, tok, goldens, system, device, epochs=1, lr=1e-5):
    """Fine-tune on pooled goldens. Returns {"loss_before", "loss_after"}.

    LoRA must already be attached (caller decides — GRPO and SFT share the
    same trainable head, so a prior `grpo.add_lora(model)` is reused).
    """
    examples = [_golden_example(g) for g in goldens]
    torch = grpo._torch()

    before = _sft_loss(model, tok, examples, system, device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for example in examples:
            ids, labs = grpo.build_masked(example, system, tok)
            t = torch.tensor([ids], device=device)
            msk = torch.tensor([[0. if l == -100 else 1. for l in labs]],
                               device=device)[:, 1:]
            logits = model(t).logits[:, :-1]
            lp = torch.log_softmax(logits.float(), -1).gather(
                -1, t[:, 1:].unsqueeze(-1)).squeeze(-1)
            ce = -(lp * msk)
            loss = ce.sum() / msk.sum().clamp(min=1)
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grpo.GRAD_CLIP)
        opt.step()

    model.eval()
    after = _sft_loss(model, tok, examples, system, device)
    return {"loss_before": round(before, 4), "loss_after": round(after, 4),
            "examples": len(examples)}