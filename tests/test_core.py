import pytest

from app.config import Settings
from app.conversation import build_case_state
from app.l2_client import L2Client
from app.mcp_client import (
    MCPRegistry,
    capture_citations,
)
from app.pipeline import HealthBenchHarness
from app.risk import assess_risk


def test_multiturn_state():
    case = build_case_state(
        [
            {
                "role": "user",
                "content": "I take aspirin.",
            },
            {
                "role": "user",
                "content": (
                    "Is it safe during pregnancy?"
                ),
            },
        ]
    )

    assert case["turns"] == 2

    assert (
        "aspirin"
        in case["conversation_text"]
    )


def test_nested_citation_capture():
    citation_store = {}

    capture_citations(
        {
            "items": [
                {
                    "cite_uid": "cite-test-1",
                    "content": "Test evidence",
                }
            ]
        },
        "test_tool",
        citation_store,
    )

    assert "cite-test-1" in citation_store

    assert (
        citation_store[
            "cite-test-1"
        ]["content"]
        == "Test evidence"
    )


def test_emergency_detection():
    case = {
        "conversation_text": (
            "I have chest pain and "
            "shortness of breath."
        )
    }

    result = assess_risk(case)

    assert result[
        "possible_emergency"
    ] is True


@pytest.mark.asyncio
async def test_mock_pipeline():
    settings = Settings()

    object.__setattr__(
        settings,
        "l2_mode",
        "mock",
    )

    # Unit tests must not contact the real MCP server.
    object.__setattr__(
        settings,
        "enable_retrieval",
        False,
    )

    registry = MCPRegistry(
        url="https://example.invalid/mcp",
        api_key="",
        timeout=10,
    )

    harness = HealthBenchHarness(
        L2Client(settings),
        registry,
        settings,
    )

    result = await harness.answer(
        [
            {
                "role": "user",
                "content": "General question",
            }
        ]
    )

    assert result.answer

    assert (
        result.retrieval_status
        == "disabled"
    )