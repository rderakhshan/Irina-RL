"""The acting provider: a drop-in replacement for `back.provider.complete` that
runs a small Qwen2.5-Coder model instead of calling the DeepSeek API.

This is the student. It is the ONLY trainable thing in the whole project, and
it must present the exact face the harness expects — `complete(model, system,
messages, tools)` returning `{"text", "tool_calls", "usage"}` — so that a
one-line swap (below) turns the harness' acting model into Qwen without
touching a single line of `src/back/`.

Unlike the DeepSeek provider, `complete` here ignores both its `temperature`
argument and its `model` argument (read module-global `TEMPERATURE`/`DEFAULT_MODEL`
instead), because the harness calls it with its own stored names: chat wants
~0.4, GRPO rollouts want ~1.0, and the harness stamps every call
`deepseek-chat` — an HTTP API name that must never be passed to a model
loader. The caller flips the temperature knob between runs.

The chat template renders the tool list into the system prompt and asks for a
JSON tool call, exactly the contract `traj.to_chat` / `traj.parse_tool_calls`
implement — so generation and training segmentation always see one canonical
message form.
"""

import os

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TORCH_OK = True
except Exception:  # pragma: no cover - offline dev box has no torch
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    _TORCH_OK = False

from . import traj

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
TEMPERATURE = 0.4           # chat default; GRPO flips this to ~1.0
MAX_OUTPUT_TOKENS = 768

_LOADED = {"model": None, "tokenizer": None, "name": None, "device": None}


def reset():
    """Forget any cached model (e.g. to reload after a LoRA save)."""
    _LOADED.update(model=None, tokenizer=None, name=None, device=None)


def load(model_name=DEFAULT_MODEL):
    """Load (and cache) the acting model. fp16 on CUDA, fp32 on CPU."""
    if _LOADED["name"] != model_name:
        if not _TORCH_OK:
            raise RuntimeError(
                "provider_local needs torch + transformers. Install them in the "
                "Colab runtime (see PLAN.md) — they are not importable here.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa").to(device)
        model.eval()
        print(f"acting provider: {model_name} on {device} "
              f"({sum(p.numel() for p in model.parameters()):,} params)")
        _LOADED.update(model=model, tokenizer=tok, name=model_name,
                       device=device)
    return _LOADED["model"], _LOADED["tokenizer"], _LOADED["device"]


def complete(model, system, messages, tools, temperature=None):
    """One acting turn. Same signature and return shape as the DeepSeek
    provider (back.provider.complete) so the harness runs unchanged.

    The `model` argument is ignored on purpose: the harness stamps every call
    with its own stored model name (`provider.DEFAULT_MODEL`, i.e.
    `deepseek-chat`), and that string must NEVER reach a model loader — it is
    an HTTP API name, not a download. This provider always acts as its own
    fixed local model.

    Returns {"text", "tool_calls": [{name, args, signature}], "usage"}.
    `signature` is always None — the harness derives call specs from the tool
    schema, not from the reply.
    """
    acting_model, tok, device = load()

    # The system prompt both carries the harness' instructions AND the tool
    # catalog, rendered as JSON. Chat sampling temperature comes from the
    # module knob, not the (ignored) argument.
    system_full = (system or "") + traj.tool_block(tools)
    chat = traj.to_chat(messages, system_full)
    prompt = tok.apply_chat_template(chat, tokenize=False,
                                     add_generation_prompt=True)

    inputs = tok(prompt, return_tensors="pt").to(device)
    temp = TEMPERATURE if temperature is None else temperature
    with torch.no_grad():
        gen = acting_model.generate(
            **inputs,
            max_new_tokens=MAX_OUTPUT_TOKENS,
            do_sample=temp > 0,
            temperature=max(temp, 1e-3) if temp > 0 else None,
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = gen[0][inputs["input_ids"].shape[1]:]
    reply = tok.decode(new_tokens, skip_special_tokens=True)

    call = traj.parse_tool_calls(reply)
    calls = [{"name": call["name"], "args": call["arguments"],
              "signature": None}] if call else []
    # When a tool call was parsed, the *text* is what the model said around it;
    # keep prose, because a reasoning model annotates before acting.
    return {"text": reply, "tool_calls": calls,
            "usage": {"input": int(inputs["input_ids"].shape[1]),
                      "output": int(new_tokens.shape[0])}}


def install_provider():
    """Swap the harness' model door to this acting provider.

    `back.loop` and `back.context` both call `provider.complete` through the
    module, so overwriting `back.provider.complete` is enough — zero edits to
    harness files. Call `back.provider.complete = provider_local.complete`
    yourself if you want to keep `back.provider.complete` around.
    """
    import back.provider as provider
    provider.complete = complete
    print("acting provider installed: harness now runs Qwen "
          f"({DEFAULT_MODEL}) via provider_local")