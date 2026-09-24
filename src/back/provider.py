"""Day 1 — the provider: the one place in the harness that talks to a model.

The concept is the *neutral message format*. The rest of the harness speaks a
small vendor-free dialect — user, assistant, tool — and this module is the only
translator between that dialect and a model's wire format.

This copy speaks to DeepSeek, whose API is OpenAI-compatible: `chat/completions`,
`messages`, `tool_call_id` round-trips, `Authorization: Bearer`. Swapping
vendors means rewriting this file and nothing else.

Design rules this file embodies:
  - One boundary. When the vendor changes their JSON, only this file changes.
  - Return plain dicts, never vendor objects: nothing above should learn an SDK.
  - Retry the transient, raise the permanent: a 429 is weather, a 400 is a bug.
"""

import json
import os
import time
import urllib.error
import urllib.request
import uuid

API_ROOT = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
MAX_OUTPUT_TOKENS = 8192  # max for deepseek-chat


def api_key():
    """Return the DeepSeek key: DEEPSEEK_API_KEY first, with the harness's
    generic ODYSSEUS_API_KEY as a fallback.
    """
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ODYSSEUS_API_KEY")
    if not key:
        raise RuntimeError("No API key found. Set DEEPSEEK_API_KEY (or "
                           "ODYSSEUS_API_KEY), e.g. in the root `.env`.")
    return key


def complete(model, system, messages, tools, temperature=0.4):
    """Send one request to the model and return a normalised reply.

    `messages` uses the neutral format described in `_to_wire`; `tools` is a
    list of spec dicts shaped {"schema": {...}}, empty for a prose-only turn.
    Returns {"text", "tool_calls": [{"name", "args", "signature"}], "usage"}.
    """
    body = {"model": model,
            "messages": _with_system(system, _to_wire(messages)),
            "temperature": temperature,
            "max_tokens": MAX_OUTPUT_TOKENS}
    if tools:
        body["tools"] = [{"type": "function", "function": t["schema"]}
                         for t in tools]
        body["tool_choice"] = "auto"
    data = _post(f"{API_ROOT}/chat/completions", body)
    message = (data.get("choices") or [{}])[0].get("message") or {}

    text = message.get("content") or ""
    calls = []
    for fc in message.get("tool_calls") or []:
        raw = (fc.get("function") or {}).get("arguments")
        try:
            args = json.loads(raw) if raw else {}
        except (ValueError, TypeError):
            # Arguments are only loosely JSON out of a model; never crash the
            # loop over it — hand the raw string back as a single value so the
            # model can see and fix its own mistake on the next turn.
            args = {"_raw": raw}
        calls.append({"name": (fc.get("function") or {}).get("name", ""),
                      "args": args,
                      "signature": None,
                      "id": fc.get("id")})
    usage = data.get("usage") or {}
    return {"text": str(text), "tool_calls": calls,
            "usage": {"input": usage.get("prompt_tokens", 0),
                      "output": usage.get("completion_tokens", 0)}}


def _with_system(system, wire):
    """OpenAI-style requests lead with a system message, when there is one."""
    if not system:
        return wire
    return [{"role": "system", "content": system}] + wire


def _to_wire(messages):
    """Translate neutral messages into OpenAI `messages`. The neutral format,
    all the rest of the harness ever writes, is {"role": "user", "text"};
    {"role": "assistant", "text", "tool_calls"}; {"role": "tool", "name",
    "text"}. OpenAI splits a tool result onto its own role that answers one
    call by `tool_call_id` — and the harness does not carry the id on the tool
    message, so this module owns the pairing: the ids walk out of each
    assistant turn in call order and into the tool turns that follow.
    """
    wire, pending = [], []
    for message in messages:
        role = message["role"]
        if role == "user":
            wire.append({"role": "user", "content": message.get("text", "")})
        elif role == "assistant":
            msg = {"role": "assistant", "content": message.get("text", "")}
            calls = message.get("tool_calls") or []
            if calls:
                msg["tool_calls"] = []
                for call in calls:
                    call_id = call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                    pending.append(call_id)
                    msg["tool_calls"].append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": call["name"],
                                     "arguments": json.dumps(
                                         call.get("args") or {},
                                         ensure_ascii=False)}})
            wire.append(msg)
        elif role == "tool":
            wire.append({"role": "tool",
                         "tool_call_id": pending.pop(0) if pending else None,
                         "content": message.get("text", "")})
    return wire


def _post(url, body, retries=5):
    """POST JSON and return the parsed reply, retrying what is transient. The
    key travels in the Authorization header, so it stays out of shell history,
    proxy logs and pasted tracebacks.
    """
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key()}"}
    for attempt in range(retries):
        req = urllib.request.Request(url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            # 429 and 5xx are the server asking us to wait; anything else is
            # our own malformed request, which retrying cannot fix.
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(f"DeepSeek HTTP {e.code}: "
                               f"{e.read().decode('utf-8', 'replace')[:400]}") from None
        except (urllib.error.URLError, TimeoutError):
            # A dropped connection is weather too; treat it like a 503.
            if attempt >= retries - 1:
                raise
            time.sleep(2 ** attempt * 2)