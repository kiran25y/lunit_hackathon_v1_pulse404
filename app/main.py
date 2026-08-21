from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import Settings
from .l2_client import L2Client
from .mcp_client import MCPRegistry
from .pipeline import HealthBenchHarness


def harness() -> HealthBenchHarness:
    settings = Settings()

    l2_client = L2Client(settings)

    mcp_registry = MCPRegistry(
        url=settings.mcp_url,
        api_key=settings.mcp_api_key,
        timeout=settings.mcp_timeout,
    )

    return HealthBenchHarness(
        l2=l2_client,
        mcp=mcp_registry,
        settings=settings,
    )

def format_exception(
    exc: BaseException,
    level: int = 0,
) -> list[dict]:
    """
    Recursively expose Python ExceptionGroup errors without
    including secrets or request headers.
    """

    output = [
        {
            "level": level,
            "type": type(exc).__name__,
            "message": str(exc),
        }
    ]

    nested = getattr(
        exc,
        "exceptions",
        None,
    )

    if nested:
        for child in nested:
            output.extend(
                format_exception(
                    child,
                    level + 1,
                )
            )

    cause = getattr(
        exc,
        "__cause__",
        None,
    )

    if cause is not None:
        output.extend(
            format_exception(
                cause,
                level + 1,
            )
        )

    return output

async def run_doctor() -> None:
    """
    Validate configuration and MCP connectivity without running
    the complete medical-answer pipeline.
    """

    settings = Settings()
    client = harness()

    result = {
        "mode": settings.l2_mode,
        "base_url_set": bool(
            settings.l2_base_url
        ),
        "model": settings.l2_model,
        "l2_api_key_set": bool(
            settings.l2_api_key
        ),
        "l2_timeout_seconds": (
            settings.l2_timeout
        ),
        "mcp_url": settings.mcp_url,
        "mcp_api_key_set": bool(
            settings.mcp_api_key
        ),
        "mcp_timeout_seconds": (
            settings.mcp_timeout
        ),
        "retrieval_enabled": (
            settings.enable_retrieval
        ),
        "revision_enabled": (
            settings.enable_revision
        ),
        "retrieval_max_calls": (
            settings.retrieval_max_calls
        ),
    }

    try:
        schemas = await client.mcp.schemas()

        tool_names = [
            schema["function"]["name"]
            for schema in schemas
        ]

        result["mcp_connected"] = True
        result["mcp_tool_count"] = len(
            tool_names
        )
        result["mcp_tools"] = tool_names

    except BaseException as exc:
        result["mcp_connected"] = False
        result["mcp_tool_count"] = 0
        result["mcp_error_type"] = (
            type(exc).__name__
        )
        result["mcp_error"] = str(exc)
        result["mcp_error_tree"] = (
            format_exception(exc)
        )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


async def run_request(
    messages: list[dict],
) -> None:
    client = harness()

    result = await client.answer(messages)

    print(
        result.model_dump_json(
            indent=2,
        )
    )


async def run(args) -> None:
    if args.command == "doctor":
        await run_doctor()
        return

    if args.command in {
        "demo",
        "smoke",
    }:
        messages = [
            {
                "role": "user",
                "content": (
                    "According to the available clinical "
                    "guidelines, what blood pressure target "
                    "is recommended for adults with chronic "
                    "kidney disease?"
                ),
            }
        ]

        await run_request(messages)
        return

    if args.command == "ask":
        if args.input:
            input_path = Path(args.input)

            if not input_path.exists():
                raise FileNotFoundError(
                    f"Input file not found: "
                    f"{input_path}"
                )

            input_data = json.loads(
                input_path.read_text(
                    encoding="utf-8",
                )
            )

            if isinstance(input_data, dict):
                messages = input_data.get(
                    "messages"
                )
            else:
                messages = input_data

            if not isinstance(messages, list):
                raise ValueError(
                    "The input file must contain a "
                    "messages array or be a messages array."
                )

        else:
            if not args.text:
                raise ValueError(
                    "Provide --text or --input."
                )

            messages = [
                {
                    "role": "user",
                    "content": args.text,
                }
            ]

        await run_request(messages)
        return

    raise ValueError(
        f"Unsupported command: {args.command}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lunit FM L2 HealthBench harness"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "doctor",
        help=(
            "Validate L2 configuration and "
            "discover MCP tools."
        ),
    )

    subparsers.add_parser(
        "demo",
        help=(
            "Run a real or mock example."
        ),
    )

    subparsers.add_parser(
        "smoke",
        help=(
            "Run an end-to-end grounded "
            "medical query."
        ),
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help=(
            "Run a custom request."
        ),
    )

    ask_parser.add_argument(
        "--text",
        default="",
        help="Single user message.",
    )

    ask_parser.add_argument(
        "--input",
        help=(
            "Path to a JSON conversation file."
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    asyncio.run(
        run(args)
    )


if __name__ == "__main__":
    main()