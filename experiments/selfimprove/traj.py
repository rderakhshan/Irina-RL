"""Transcript conversion: a neutral harness transcript is the RL training
example, but torch's chat template needs OpenAI-ish chat dicts. This module is
the only translator between the two, and it is shared by the acting provider
(`provider_local.complete` renders through it) and the trainer
(`grpo.build_masked` segments with it) so masking and generation always agree.

Roles map like this:

    neutral ("user", "assistant", "tool")  ->  chat (system/user/assistant)

A neutral assistant message can carry `text` *and* `tool_calls`; the chat form
renders the calls as JSON text so the whole turn is one contiguous block the
trainer can label. A neutral tool result becomes a `user` message with a
`[tool <name>]` prefix, which keeps the chat template append-only — the
property `grpo.build_masked` asserts on. The system prompt is prepended, never
stored in the transcript itself.
"""

import json

SYSTEM_TAIL = """

You are an autonomous coding agent. When you want to use a tool, reply with ONE
JSON object on its own, exactly in the form {"name": "<tool>", "arguments":
{...}}, and nothing else in that message. When you are done, reply in plain
text. Every previous tool result is shown to you as `[tool <name>]\n<output>`.
"""


def _as_chat_message(message):
    """Translate one neutral message into a Qwen-style chat dict."""
    role = message["role"]
    if role == "user":
        return {"role": "user", "content": message.get("text", "")}
    if role == "tool":
        return {"role": "user",
                "content": f"[tool {message.get('name', '?')}]\n"
                           f"{message.get('text', '')}"}
    if role == "assistant":
        content = message.get("text", "") or ""
        calls = message.get("tool_calls") or []
        if calls:
            rendered = []
            for call in calls:
                rendered.append(json.dumps(
                    {"name": call["name"], "arguments": call.get("args") or {}},
                    ensure_ascii=False))
            content = (content + "\n" if content else "") + "\n".join(rendered)
        return {"role": "assistant", "content": content}
    raise ValueError(f"unknown neutral role {role!r}")


def to_chat(messages, system=""):
    """Neutral messages -> [system?] + chat dicts, in the exact order the
    harness produced them. `system` is supplied separately by the caller
    (provider / trainer) and is not part of the transcript.
    """
    chat = []
    if system:
        chat.append({"role": "system", "content": system + SYSTEM_TAIL})
    for message in messages:
        chat.append(_as_chat_message(message))
    return chat


def tool_block(tools):
    """Render the harness `tools` (specs with {"schema": {..}}) as the JSON
    tool list the acting model is told about. Appended to the system prompt so
    the model knows what it may call and in what shape.
    """
    if not tools:
        return "\n\nAvailable tools: none."
    lines = []
    for spec in tools:
        schema = spec.get("schema") or {}
        desc = schema.get("description", "") or ""
        params = schema.get("parameters") or {}
        properties = params.get("properties") or {}
        args = {name: (s or {}).get("description", "")
                for name, s in sorted(properties.items())}
        required = params.get("required") or list(properties.keys())
        lines.append(json.dumps({
            "name": schema.get("name", "?"),
            "description": desc,
            "arguments": args,
            "required": required,
        }, ensure_ascii=False))
    return "\n\nAvailable tools:\n" + "\n".join(lines)


def parse_tool_calls(text):
    """Pull one tool invocation out of a reply. The model is told to emit a
    single JSON object {"name": ..., "arguments": {...}}; we accept it even if
    it arrives wrapped in a ```json fence or alongside prose. Scans for a
    balanced-brace JSON object that carries `name` plus `arguments`, so nested
    objects in `arguments` do not truncate the match.
    """
    if not text:
        return None
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
            if start < 0:
                start = i
        elif ch == "}":
            depth -= 1
            if start >= 0 and depth == 0:
                obj = _parse_obj(text[start:i + 1])
                if obj is not None:
                    return obj
                start = -1
    return None


def _parse_obj(raw):
    """Try to parse `raw` as a tool-call object; None when it is not one."""
    try:
        obj = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(obj, dict) or "name" not in obj or "arguments" not in obj:
        return None
    args = obj["arguments"]
    if not isinstance(args, dict):
        try:
            args = {"_raw": args}
        except TypeError:
            args = {}
    return {"name": obj["name"], "arguments": args}