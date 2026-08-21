#!/usr/bin/env python3
"""
l2_client.py — the real L2 client. Stdlib only.

Configure with environment variables, then use it anywhere the harness
expects an L2Client:

    export L2_BASE_URL="https://..."      # from the help desk
    export L2_API_KEY="..."
    export L2_MODEL="l2"                  # whatever `probe_l2.py` showed

    from l2_client import RealL2
    client = RealL2()

If the endpoint is OpenAI-compatible (most internal LLM gateways are),
this works unchanged. If it isn't, `probe_l2.py` tells you what it wants
and you adjust three env vars rather than editing code:

    L2_CHAT_PATH=/v1/chat/completions     # path appended to base url
    L2_AUTH_HEADER=Authorization          # or api-key, x-api-key
    L2_AUTH_PREFIX="Bearer "              # set to "" for raw-key headers

Response parsing handles three shapes so you are not stuck if the gateway
is Anthropic-style or emits tool calls as JSON inside the text.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request

log = logging.getLogger("l2.client")

# CoEval evaluates requests concurrently. Each synchronous FastAPI request
# remains on one worker thread, so a thread-local deadline isolates budgets.
_tls = threading.local()


def set_deadline(seconds_from_now: float) -> None:
    _tls.deadline = time.monotonic() + seconds_from_now


def _remaining() -> float:
    deadline = getattr(_tls, "deadline", None)
    return float("inf") if deadline is None else deadline - time.monotonic()

BASE = os.environ.get("L2_BASE_URL", "").rstrip("/")
KEY = os.environ.get("L2_API_KEY", "")
MODEL = os.environ.get("L2_MODEL", "l2")
CHAT_PATH = os.environ.get("L2_CHAT_PATH", "/v1/chat/completions")
AUTH_HEADER = os.environ.get("L2_AUTH_HEADER", "Authorization")
AUTH_PREFIX = os.environ.get("L2_AUTH_PREFIX", "Bearer ")
HTTP_TIMEOUT = int(os.environ.get("L2_TIMEOUT", "45"))


class RealL2:
    """Duck-types l2_harness.L2Client. Never raises out of .call()."""

    def __init__(self, base_url: str = "", api_key: str = "", model: str = ""):
        self.base = (base_url or BASE).rstrip("/")
        self.key = api_key or KEY
        self.model = model or MODEL
        if not self.base:
            raise RuntimeError("L2_BASE_URL is not set")

    # -- transport --------------------------------------------------------

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.key:
            h[AUTH_HEADER] = f"{AUTH_PREFIX}{self.key}"
        return h

    @staticmethod
    def set_deadline(seconds_from_now: float) -> None:
        set_deadline(seconds_from_now)

    def _post(self, payload: dict) -> dict:
        remaining = _remaining()
        if remaining < 3:
            raise TimeoutError("request budget exhausted before L2 call")

        req = urllib.request.Request(
            self.base + CHAT_PATH,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        timeout = min(HTTP_TIMEOUT, max(3.0, remaining - 2.0))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))

    # -- api --------------------------------------------------------------

    def chat(self, system, messages, tools=None, temperature=0.2, max_tokens=2048) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + _clean(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [
                {"type": "function",
                 "function": {"name": t["name"],
                              "description": t.get("description", ""),
                              "parameters": t.get("parameters") or {"type": "object", "properties": {}}}}
                for t in tools
            ]
            payload["tool_choice"] = "auto"
        return _parse(self._post(payload))

    def call(self, **kw) -> dict:
        """Retry wrapper. Mirrors l2_harness.L2Client.call."""
        retries = int(os.environ.get("L2_RETRIES", "1"))
        last = None
        for attempt in range(retries + 1):
            try:
                return self.chat(**kw)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:300]
                last = f"HTTP {e.code}: {body}"
                if e.code in (400, 401, 403, 404):
                    log.error("L2 %s — not retrying", last)
                    break
            except Exception as e:  # noqa: BLE001
                last = str(e)[:300]
            if _remaining() < 8:
                log.error("L2 call failed and request budget is spent: %s", last)
                break
            log.warning("L2 call failed (attempt %d): %s", attempt + 1, last)
            time.sleep(1.0)
        log.error("L2 call gave up: %s", last)
        return {"content": "", "tool_calls": []}


def _clean(messages: list[dict]) -> list[dict]:
    """Some gateways reject unknown roles. Map 'tool' to a user turn if so."""
    if os.environ.get("L2_NO_TOOL_ROLE"):
        out = []
        for m in messages:
            if m.get("role") == "tool":
                out.append({"role": "user", "content": f"[tool result]\n{m['content']}"})
            else:
                out.append(m)
        return out
    return messages


def _parse(data: dict) -> dict:
    """Normalise OpenAI / Anthropic / text-embedded tool calls to one shape."""
    content, calls = "", []

    # --- OpenAI style -----------------------------------------------------
    choices = data.get("choices")
    if choices:
        msg = choices[0].get("message") or choices[0].get("delta") or {}
        content = msg.get("content") or ""
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            calls.append({"name": fn.get("name", ""),
                          "arguments": _loads(fn.get("arguments"))})
        if not calls and msg.get("function_call"):
            fc = msg["function_call"]
            calls.append({"name": fc.get("name", ""), "arguments": _loads(fc.get("arguments"))})

    # --- Anthropic style --------------------------------------------------
    elif isinstance(data.get("content"), list):
        for block in data["content"]:
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                calls.append({"name": block.get("name", ""), "arguments": block.get("input") or {}})

    # --- plain completion -------------------------------------------------
    elif isinstance(data.get("content"), str):
        content = data["content"]
    elif data.get("text"):
        content = data["text"]
    elif data.get("output"):
        content = str(data["output"])

    # --- last resort: a tool call the model wrote as JSON in the text -----
    if not calls and content:
        embedded = _extract_embedded_call(content)
        if embedded:
            calls.append(embedded)

    return {"content": content or "", "tool_calls": calls}


def _loads(raw):
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


_CALL_RE = re.compile(
    r'\{[^{}]*"(?:name|tool|tool_name|function)"\s*:\s*"([A-Za-z0-9_]+)"'
    r'[^{}]*"(?:arguments|args|parameters|input)"\s*:\s*(\{.*?\})',
    re.S,
)

_LUNIT_CALL_RE = re.compile(
    r"<tool_call>\s*([A-Za-z0-9_]+)\s*\(\s*(\{.*?\})\s*\)"
    r"\s*</arg_value>",
    re.S,
)


def _extract_embedded_call(text: str):
    """Some models emit tool calls as JSON in the message body. Rescue it."""
    native = _LUNIT_CALL_RE.search(text)
    if native:
        return {
            "name": native.group(1),
            "arguments": _loads(native.group(2)),
        }

    m = _CALL_RE.search(text)
    if not m:
        return None
    return {"name": m.group(1), "arguments": _loads(m.group(2))}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
    c = RealL2()
    print(f"base:  {c.base}{CHAT_PATH}")
    print(f"model: {c.model}\n")
    out = c.call(
        system="You are a medical expert. Answer in one sentence.",
        messages=[{"role": "user", "content": "What is the first-line treatment for uncomplicated hypertension?"}],
        temperature=0.0,
        max_tokens=120,
    )
    print("content:   ", (out["content"] or "(empty)")[:400])
    print("tool_calls:", out["tool_calls"])
    if not out["content"]:
        print("\nEmpty. Run `python3 probe_l2.py` to find the right path/auth/payload,")
        print("then set L2_CHAT_PATH / L2_AUTH_HEADER / L2_AUTH_PREFIX to match.")
