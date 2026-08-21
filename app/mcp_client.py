from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import (
    streamable_http_client,
)


class MCPRegistry:
    """
    Client for Lunit's authenticated Streamable HTTP MCP server.

    The registry dynamically discovers available tool schemas and
    executes tools through the MCP protocol.
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        timeout: float = 60,
    ):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout

        self.by_name: dict[
            str,
            dict[str, Any],
        ] = {}

        self._schemas: list[
            dict[str, Any]
        ] = []

    def _headers(self) -> dict[str, str]:
        """
        Construct authentication headers.

        MCP 1.29.0 receives these through an httpx.AsyncClient.
        The MCP SDK manages Accept and protocol-specific headers.
        """

        headers = {}

        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        return headers

    def _http_timeout(
        self,
    ) -> httpx.Timeout:
        """
        Create an HTTP timeout configuration for MCP requests.
        """

        return httpx.Timeout(
            connect=30.0,
            read=self.timeout,
            write=self.timeout,
            pool=30.0,
        )

    def _session_timeout(
        self,
    ) -> timedelta:
        """
        Create the timeout used by the MCP ClientSession.
        """

        return timedelta(
            seconds=self.timeout
        )

    async def discover(
        self,
    ) -> list[dict[str, Any]]:
        """
        Connect to Lunit MCP, initialize a session, and discover tools.
        """

        async with httpx.AsyncClient(
            headers=self._headers(),
            timeout=self._http_timeout(),
            follow_redirects=True,
        ) as http_client:
            async with streamable_http_client(
                self.url,
                http_client=http_client,
            ) as streams:
                read_stream = streams[0]
                write_stream = streams[1]

                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=(
                        self._session_timeout()
                    ),
                ) as session:
                    await session.initialize()

                    tools_result = (
                        await session.list_tools()
                    )

        schemas: list[dict[str, Any]] = []
        by_name: dict[
            str,
            dict[str, Any],
        ] = {}

        for tool in tools_result.tools:
            tool_data = tool.model_dump(
                by_alias=True,
                exclude_none=True,
            )

            name = tool_data["name"]

            input_schema = tool_data.get(
                "inputSchema"
            )

            if input_schema is None:
                input_schema = tool_data.get(
                    "input_schema",
                    {
                        "type": "object",
                        "properties": {},
                    },
                )

            by_name[name] = tool_data

            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": (
                            tool_data.get(
                                "description",
                                "",
                            )
                        ),
                        "parameters": (
                            input_schema
                        ),
                    },
                }
            )

        self.by_name = by_name
        self._schemas = schemas

        return schemas

    async def schemas(
        self,
    ) -> list[dict[str, Any]]:
        """
        Return cached tool schemas, discovering them on first use.
        """

        if not self._schemas:
            await self.discover()

        return self._schemas

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute one Lunit MCP tool and return serialized output.
        """

        if not self.by_name:
            await self.discover()

        if name not in self.by_name:
            raise KeyError(
                f"Unknown Lunit MCP tool: {name}"
            )

        async with httpx.AsyncClient(
            headers=self._headers(),
            timeout=self._http_timeout(),
            follow_redirects=True,
        ) as http_client:
            async with streamable_http_client(
                self.url,
                http_client=http_client,
            ) as streams:
                read_stream = streams[0]
                write_stream = streams[1]

                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=(
                        self._session_timeout()
                    ),
                ) as session:
                    await session.initialize()

                    result = await session.call_tool(
                        name,
                        arguments=arguments,
                    )

        result_data = result.model_dump(
            by_alias=True,
            exclude_none=True,
        )

        if result_data.get(
            "isError",
            result_data.get(
                "is_error",
                False,
            ),
        ):
            error_text = extract_text(
                result_data
            )

            raise RuntimeError(
                error_text
                or (
                    f"MCP tool {name} "
                    "returned an error."
                )
            )

        return result_data


def extract_text(
    result: dict[str, Any],
) -> str:
    """
    Extract readable text from MCP content blocks.
    """

    texts = []

    for content_item in result.get(
        "content",
        [],
    ):
        if not isinstance(
            content_item,
            dict,
        ):
            continue

        if content_item.get("type") != "text":
            continue

        text = content_item.get("text")

        if text:
            texts.append(text)

    return "\n".join(texts)


def try_parse_json(
    value: str,
) -> Any | None:
    """
    Attempt to parse a string containing serialized JSON.
    """

    try:
        return json.loads(value)

    except (
        json.JSONDecodeError,
        TypeError,
    ):
        return None


def capture_citations(
    obj: Any,
    tool_name: str,
    store: dict[str, dict[str, Any]],
) -> None:
    """
    Recursively capture citable evidence.

    Lunit tool results may carry cite_uid inside:

    - structuredContent;
    - ordinary nested dictionaries;
    - JSON serialized inside an MCP TextContent block.
    """

    if isinstance(obj, dict):
        cite_uid = obj.get("cite_uid")

        if cite_uid:
            raw_content = obj.get(
                "content",
                obj.get(
                    "text",
                    obj.get(
                        "page_content",
                        obj.get(
                            "abstract",
                            "",
                        ),
                    ),
                ),
            )

            if isinstance(
                raw_content,
                str,
            ):
                content = raw_content

            else:
                content = json.dumps(
                    raw_content,
                    ensure_ascii=False,
                )

            source_type = obj.get(
                "source_type",
                obj.get(
                    "corpus_tag",
                    obj.get(
                        "data_source",
                        "unknown",
                    ),
                ),
            )

            title = obj.get(
                "title",
                obj.get(
                    "document_title",
                    obj.get(
                        "drug_name",
                        "",
                    ),
                ),
            )

            url = obj.get(
                "url",
                obj.get(
                    "source_url",
                    obj.get(
                        "link",
                        "",
                    ),
                ),
            )

            store[cite_uid] = {
                "cite_uid": cite_uid,
                "source_type": source_type,
                "title": title,
                "url": url,
                "content": content,
                "tool_name": tool_name,
            }

        # Some MCP tools return structured JSON encoded inside the
        # text field of a TextContent result.
        if (
            obj.get("type") == "text"
            and isinstance(
                obj.get("text"),
                str,
            )
        ):
            parsed = try_parse_json(
                obj["text"]
            )

            if parsed is not None:
                capture_citations(
                    parsed,
                    tool_name,
                    store,
                )

        for value in obj.values():
            capture_citations(
                value,
                tool_name,
                store,
            )

    elif isinstance(obj, list):
        for value in obj:
            capture_citations(
                value,
                tool_name,
                store,
            )