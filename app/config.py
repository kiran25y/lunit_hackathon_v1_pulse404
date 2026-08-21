from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _number(
    name: str,
    default: int | float,
    converter,
):
    """Read numeric settings without making server startup fragile."""

    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        return converter(raw_value)
    except (TypeError, ValueError):
        return default


def _bool(
    name: str,
    default: bool,
) -> bool:
    return os.getenv(
        name,
        str(default),
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class Settings:
    # Lunit FM L2
    l2_mode: str = os.getenv(
        "L2_MODE",
        "mock",
    )

    l2_base_url: str = os.getenv(
        "L2_BASE_URL",
        os.getenv(
            "LUNIT_FM_API_URL",
            "",
        ),
    )

    l2_api_key: str = os.getenv(
        "LUNIT_FM_API_KEY",
        os.getenv(
            "L2_API_KEY",
            "",
        ),
    )

    l2_model: str = os.getenv(
        "L2_MODEL",
        os.getenv(
            "LUNIT_FM_MODEL",
            "Lunit/L2-preview",
        ),
    )

    l2_timeout: float = _number(
        "L2_TIMEOUT_SECONDS",
        300.0,
        float,
    )

    # Backwards-compatible alias used by l2_client.py.
    @property
    def timeout(self) -> float:
        return self.l2_timeout

    # Lunit MCP server
    mcp_url: str = os.getenv(
        "LUNIT_MCP_URL",
        "https://mcp.hackathon.lunit.io/mcp",
    )

    mcp_api_key: str = os.getenv(
        "LUNIT_FM_API_KEY",
        os.getenv(
            "L2_API_KEY",
            "",
        ),
    )

    mcp_timeout: float = _number(
        "LUNIT_MCP_TIMEOUT_SECONDS",
        60.0,
        float,
    )

    # Harness configuration
    retrieval_max_calls: int = _number(
        "RETRIEVAL_MAX_CALLS",
        8,
        int,
    )

    max_selected_citations: int = _number(
        "MAX_SELECTED_CITATIONS",
        6,
        int,
    )

    enable_retrieval: bool = _bool(
        "ENABLE_RETRIEVAL",
        True,
    )

    enable_revision: bool = _bool(
        "ENABLE_REVISION",
        True,
    )

    trace_dir: str = os.getenv(
        "TRACE_DIR",
        "runs/traces",
    )
