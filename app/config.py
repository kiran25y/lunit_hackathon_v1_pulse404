from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _first_env(*names: str, default: str = "") -> str:
    """Return the first non-empty value from equivalent env names."""

    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()

    return default


def _number(
    names: str | tuple[str, ...],
    default: int | float,
    converter,
):
    """Read numeric settings without making server startup fragile."""

    if isinstance(names, str):
        names = (names,)

    raw_value = _first_env(*names)

    if not raw_value:
        return default

    try:
        return converter(raw_value)
    except (TypeError, ValueError):
        return default


def _bool(
    names: str | tuple[str, ...],
    default: bool,
) -> bool:
    if isinstance(names, str):
        names = (names,)

    return _first_env(
        *names,
        default=str(default),
    ).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class Settings:
    # Lunit FM L2
    l2_base_url: str = _first_env(
        "L2_BASE_URL",
        "LUNIT_FM_API_URL",
    )

    # The official environment uses LUNIT_L2_MODE. If a real endpoint is
    # supplied without an explicit mode, selecting it is safer than silently
    # returning the development mock response for every benchmark sample.
    l2_mode: str = _first_env(
        "L2_MODE",
        "LUNIT_L2_MODE",
        default=(
            "openai_compatible"
            if l2_base_url
            else "mock"
        ),
    )

    l2_api_key: str = _first_env(
        "L2_API_KEY",
        "LUNIT_FM_API_KEY",
    )

    l2_model: str = _first_env(
        "L2_MODEL",
        "LUNIT_FM_MODEL",
        default="Lunit/L2-preview",
    )

    l2_timeout: float = _number(
        (
            "L2_TIMEOUT_SECONDS",
            "LUNIT_L2_TIMEOUT",
        ),
        300.0,
        float,
    )

    # Backwards-compatible alias used by l2_client.py.
    @property
    def timeout(self) -> float:
        return self.l2_timeout

    # Lunit MCP server
    mcp_url: str = _first_env(
        "LUNIT_MCP_URL",
        default="https://mcp.hackathon.lunit.io/mcp",
    )

    mcp_api_key: str = _first_env(
        "LUNIT_FM_API_KEY",
        "L2_API_KEY",
    )

    mcp_timeout: float = _number(
        (
            "LUNIT_MCP_TIMEOUT_SECONDS",
            "LUNIT_MCP_TIMEOUT",
        ),
        60.0,
        float,
    )

    # Harness configuration
    retrieval_max_calls: int = _number(
        (
            "RETRIEVAL_MAX_CALLS",
            "LUNIT_RETRIEVAL_MAX_CALLS",
        ),
        8,
        int,
    )

    max_selected_citations: int = _number(
        "MAX_SELECTED_CITATIONS",
        6,
        int,
    )

    enable_retrieval: bool = _bool(
        (
            "ENABLE_RETRIEVAL",
            "LUNIT_ENABLE_RETRIEVAL",
        ),
        True,
    )

    enable_revision: bool = _bool(
        (
            "ENABLE_REVISION",
            "LUNIT_ENABLE_REVISION",
        ),
        True,
    )

    trace_dir: str = os.getenv(
        "TRACE_DIR",
        "runs/traces",
    )
