from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .main import harness as build_harness


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s "
        "%(name)s %(message)s"
    ),
)

logger = logging.getLogger(
    "pulse404-server"
)

DRIVER_MODEL_NAME = (
    "pulse404-healthbench"
)


class ChatMessage(BaseModel):
    role: str
    content: Any = None

    # Preserve optional fields if the evaluator sends them.
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    model_config = {
        "extra": "allow",
    }


class ChatCompletionRequest(BaseModel):
    model: str = DRIVER_MODEL_NAME
    messages: list[ChatMessage] = Field(
        default_factory=list
    )

    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False

    model_config = {
        "extra": "allow",
    }


app = FastAPI(
    title="Pulse404 Lunit HealthBench Driver",
    version="1.0.0",
)

# Reuse the registry so discovered MCP schemas can be cached.
healthbench_harness = build_harness()


@app.get("/")
async def root():
    return {
        "status": "ok",
        "model": DRIVER_MODEL_NAME,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": DRIVER_MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "pulse404",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
):
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        "Streaming is not supported."
                    ),
                    "type": (
                        "invalid_request_error"
                    ),
                    "code": (
                        "streaming_not_supported"
                    ),
                }
            },
        )

    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        "messages must contain "
                        "at least one conversation turn"
                    ),
                    "type": (
                        "invalid_request_error"
                    ),
                    "code": "empty_messages",
                }
            },
        )

    messages = [
        message.model_dump(
            exclude_none=True,
        )
        for message in request.messages
    ]

    request_id = (
        f"chatcmpl-{uuid.uuid4().hex}"
    )

    started = time.time()

    try:
        result = (
            await healthbench_harness.answer(
                messages
            )
        )

    except Exception:
        logger.exception(
            "HealthBench request failed"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": (
                        "The medical response "
                        "pipeline failed."
                    ),
                    "type": "server_error",
                    "code": "pipeline_error",
                }
            },
        )

    elapsed = round(
        time.time() - started,
        3,
    )

    logger.info(
        "request=%s turns=%s "
        "retrieval=%s evidence=%s "
        "latency=%s",
        request_id,
        len(messages),
        result.retrieval_status,
        len(result.evidence),
        elapsed,
    )

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": DRIVER_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.answer,
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