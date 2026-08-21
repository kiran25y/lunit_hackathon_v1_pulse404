import os
import time
import uuid
import logging
import threading
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

import l2_harness as H


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

log = logging.getLogger("submission")

app = FastAPI()


# ============================================================
# OpenAI-compatible request models
# ============================================================

class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = ""


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[Message]
    stream: bool | None = False


# ============================================================
# Lazy infrastructure
# ============================================================

def build_real():
    """
    Construct the official Lunit adapters only when an actual
    chat request arrives.

    This keeps GET /v1/models startup-safe.
    """

    # Bridge hackathon env vars to the official L2 client.
    #
    # Local development used LUNIT_FM_*.
    # If the evaluator provides L2_* directly, preserve them.

    if not os.environ.get("L2_BASE_URL"):
        model_url = os.environ.get(
            "LUNIT_FM_API_URL",
            "https://model.hackathon.lunit.io",
        )
        os.environ["L2_BASE_URL"] = model_url.rstrip("/")

    if not os.environ.get("L2_API_KEY"):
        key = os.environ.get("LUNIT_FM_API_KEY", "")
        if key:
            os.environ["L2_API_KEY"] = key

    if not os.environ.get("L2_MODEL"):
        os.environ["L2_MODEL"] = os.environ.get(
            "LUNIT_FM_MODEL",
            "Lunit/L2-preview",
        )

    if not os.environ.get("L2_CHAT_PATH"):
        base_url = os.environ["L2_BASE_URL"].rstrip("/")
        os.environ["L2_CHAT_PATH"] = (
            "/chat/completions"
            if base_url.endswith("/v1")
            else "/v1/chat/completions"
        )

    from l2_client import RealL2
    from lunit_mcp import LunitMCP

    return RealL2(), LunitMCP()


# Reuse discovery metadata across concurrent evaluation requests. Rebuild the
# adapters after a pipeline-level failure so a broken MCP session is not kept.
_infra_lock = threading.Lock()
_infra: list = [None]


def get_infra():
    with _infra_lock:
        if _infra[0] is None:
            _infra[0] = build_real()
        return _infra[0]


def reset_infra():
    with _infra_lock:
        _infra[0] = None


# ============================================================
# Message normalization
# ============================================================

def normalize_content(content: Any) -> str:
    """
    Accept normal OpenAI string content as well as structured
    content arrays.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        pieces = []

        for item in content:
            if isinstance(item, str):
                pieces.append(item)

            elif isinstance(item, dict):
                text = item.get("text")

                if isinstance(text, str):
                    pieces.append(text)

                elif isinstance(
                    item.get("content"),
                    str,
                ):
                    pieces.append(item["content"])

        return "\n".join(pieces)

    if content is None:
        return ""

    return str(content)


def normalize_messages(
    messages: list[Message],
) -> list[dict]:

    result = []

    for message in messages:

        role = message.role

        if role not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            role = "user"

        result.append(
            {
                "role": role,
                "content": normalize_content(
                    message.content
                ),
            }
        )

    return result


# ============================================================
# Absolute last-resort response
# ============================================================

FALLBACK_TEXT = (
    "현재 요청을 처리하는 과정에서 일부 근거 조회에 문제가 "
    "발생했습니다. 제공된 정보만으로 확정적인 판단을 내리기보다 "
    "증상이나 상황이 지속되거나 악화되면 담당 의료진과 상의하시기 "
    "바랍니다."
)


def answer_safely(messages: list[dict]) -> str:
    """
    Official harness first.

    Nothing is allowed to propagate out of this function and
    an empty response is never returned.
    """

    try:
        client, tools = get_infra()

        answer = H.answer(
            client,
            tools,
            messages,
        )

        if isinstance(answer, str) and answer.strip():
            return answer.strip()

        log.error(
            "Harness returned empty response."
        )

    except Exception:
        log.exception(
            "Harness failed; returning safe fallback."
        )
        reset_infra()

    # Never return empty and never crash the benchmark.
    return FALLBACK_TEXT


# ============================================================
# Required API
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "lunit-hackathon-agent",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "lunit-hackathon-agent",
                "object": "model",
                "created": 0,
                "owned_by": "team",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):

    started = time.time()

    try:
        messages = normalize_messages(
            request.messages
        )

        if not messages:
            answer = FALLBACK_TEXT
        else:
            answer = answer_safely(messages)

    except Exception:
        log.exception(
            "Unexpected API-layer failure."
        )
        answer = FALLBACK_TEXT

    now = int(time.time())

    response = {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": now,
        "model": request.model
        or "lunit-hackathon-agent",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

    log.info(
        "completion finished in %.2fs",
        time.time() - started,
    )

    return response
