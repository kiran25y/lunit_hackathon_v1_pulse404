import importlib

import pytest

from app.config import Settings, _normalize_l2_base_url, _number
from app.conversation import build_case_state
from app.l2_client import L2Client
from app.mcp_client import (
    MCPRegistry,
    capture_citations,
)
from app.pipeline import HealthBenchHarness, _normalize_citations
from app.risk import assess_risk


def test_invalid_numeric_environment_values_use_safe_defaults(monkeypatch):
    monkeypatch.setenv("BROKEN_NUMBER", "")
    assert _number("BROKEN_NUMBER", 60.0, float) == 60.0

    monkeypatch.setenv("BROKEN_NUMBER", "not-a-number")
    assert _number("BROKEN_NUMBER", 8, int) == 8


def test_l2_base_url_is_normalized_once():
    assert (
        _normalize_l2_base_url("https://model.example")
        == "https://model.example/v1"
    )


def test_superscript_citations_are_normalized_without_changing_units():
    answer = "The target uses m². This is supported.¹\n¹ Guideline"
    assert _normalize_citations(answer) == (
        "The target uses m². This is supported.[1]\n[1] Guideline"
    )
    assert (
        _normalize_l2_base_url("https://model.example/v1/")
        == "https://model.example/v1"
    )


def test_official_lunit_environment_selects_real_pipeline(monkeypatch):
    monkeypatch.delenv("L2_MODE", raising=False)
    monkeypatch.delenv("L2_BASE_URL", raising=False)
    monkeypatch.delenv("ENABLE_RETRIEVAL", raising=False)
    monkeypatch.setenv("LUNIT_FM_API_URL", "https://model.example/v1")
    monkeypatch.setenv("LUNIT_L2_MODE", "openai_compatible")
    monkeypatch.setenv("LUNIT_ENABLE_RETRIEVAL", "true")
    monkeypatch.setenv("LUNIT_L2_TIMEOUT", "123")

    import app.config as config_module

    reloaded = importlib.reload(config_module)
    settings = reloaded.Settings()

    assert settings.l2_mode == "openai_compatible"
    assert settings.l2_base_url == "https://model.example/v1"
    assert settings.enable_retrieval is True
    assert settings.l2_timeout == 123.0

    importlib.reload(config_module)


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


@pytest.mark.asyncio
async def test_pipeline_degrades_gracefully_when_mcp_is_unavailable():
    settings = Settings()
    object.__setattr__(settings, "l2_mode", "mock")
    object.__setattr__(settings, "enable_retrieval", True)

    class UnavailableRegistry:
        async def schemas(self):
            raise RuntimeError("MCP unavailable")

    harness = HealthBenchHarness(
        L2Client(settings),
        UnavailableRegistry(),
        settings,
    )

    result = await harness.answer(
        [{"role": "user", "content": "General question"}]
    )

    assert result.answer
    assert result.retrieval_status == "no_evidence"
    assert result.trace["retrieval"][0]["event"] == "mcp_discovery_failed"
