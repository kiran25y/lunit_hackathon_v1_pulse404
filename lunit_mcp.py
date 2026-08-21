#!/usr/bin/env python3
"""
lunit_mcp.py — a direct Streamable-HTTP MCP client for the Lunit hackathon.

Stdlib only. No pip install, no Codex, no MCP SDK.

It is two things at once:

  1. A CLI that replaces Codex's /mcp panel, so you can explore the tools
     by hand while your editor is broken:

         export LUNIT_FM_API_KEY="lunit_..."
         python3 lunit_mcp.py list
         python3 lunit_mcp.py schema index_get_relevant_nodes
         python3 lunit_mcp.py call kcd_search_codes '{"name":"type 2 diabetes"}'
         python3 lunit_mcp.py citecheck        # which tools emit cite_uid?
         python3 lunit_mcp.py dump-assets      # build the static asset cache

  2. The MCPTools implementation the harness needs:

         from lunit_mcp import LunitMCP
         tools = LunitMCP()
         H.answer(client, tools, messages)

`citecheck` is the one to run first. It answers empirically the question the
whole retrieval design hangs on: which tools return a cite_uid, and which
return values that finalize_retrieval cannot carry.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

MCP_URL = os.environ.get("LUNIT_MCP_URL", "https://mcp.hackathon.lunit.io/mcp")
API_KEY = os.environ.get("LUNIT_FM_API_KEY", "")
TIMEOUT = int(os.environ.get("LUNIT_MCP_TIMEOUT", "20"))
PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]


class MCPError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------


class StreamableHTTPClient:
    """Minimal JSON-RPC-over-HTTP MCP client.

    Handles the two response encodings a Streamable HTTP server may use:
    a plain JSON body, or an SSE stream of `data:` lines. Also carries the
    Mcp-Session-Id header once the server issues one.
    """

    def __init__(self, url: str = MCP_URL, key: str = API_KEY, timeout: int = TIMEOUT):
        if not key:
            raise MCPError("LUNIT_FM_API_KEY is not set")
        self.url, self.key, self.timeout = url, key, timeout
        self.session_id: str | None = None
        self.protocol: str = PROTOCOL_VERSIONS[0]
        self._next_id = 0
        self._initialized = False

    # -- low level --------------------------------------------------------

    def _headers(self) -> dict:
        h = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        if self._initialized:
            h["MCP-Protocol-Version"] = self.protocol
        return h

    def _post(self, payload: dict, expect_response: bool = True):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self.session_id = sid
                raw = resp.read().decode("utf-8", "replace")
                ctype = (resp.headers.get("Content-Type") or "").lower()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise MCPError(f"HTTP {e.code} {e.reason}: {detail}") from None
        except urllib.error.URLError as e:
            raise MCPError(f"cannot reach {self.url}: {e.reason}") from None

        if not expect_response:
            return None
        return _decode(raw, ctype)

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        msg = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            msg["params"] = params
        result = self._post(msg)
        if result is None:
            raise MCPError(f"{method}: empty response")
        if "error" in result:
            err = result["error"]
            raise MCPError(f"{method}: [{err.get('code')}] {err.get('message')}")
        return result.get("result", {})

    def _notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        try:
            self._post(msg, expect_response=False)
        except MCPError as e:
            # Notifications are fire-and-forget; a 202 with no body is normal.
            if "HTTP 4" in str(e) or "HTTP 5" in str(e):
                raise

    # -- handshake --------------------------------------------------------

    def initialize(self) -> dict:
        last = None
        for version in PROTOCOL_VERSIONS:
            try:
                info = self._rpc("initialize", {
                    "protocolVersion": version,
                    "capabilities": {},
                    "clientInfo": {"name": "lunit-harness", "version": "0.1"},
                })
                self.protocol = info.get("protocolVersion", version)
                self._initialized = True
                self._notify("notifications/initialized")
                return info
            except MCPError as e:
                last = e
                continue
        raise MCPError(f"initialize failed on every protocol version: {last}")

    def ensure(self) -> None:
        if not self._initialized:
            self.initialize()

    # -- api --------------------------------------------------------------

    def list_tools(self) -> list[dict]:
        self.ensure()
        tools, cursor = [], None
        while True:
            params = {"cursor": cursor} if cursor else {}
            res = self._rpc("tools/list", params)
            tools.extend(res.get("tools", []))
            cursor = res.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict) -> object:
        self.ensure()
        res = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if res.get("isError"):
            raise MCPError(f"{name} returned isError: {_flatten(res)[:300]}")
        # Prefer structured output when the server provides it.
        if "structuredContent" in res:
            return res["structuredContent"]
        return _unwrap_content(res.get("content", []))


def _decode(raw: str, ctype: str) -> dict | None:
    """Accept either a JSON body or an SSE stream."""
    raw = raw.strip()
    if not raw:
        return None
    if "text/event-stream" in ctype or raw.startswith("event:") or raw.startswith("data:"):
        last = None
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                chunk = line[5:].strip()
                if chunk and chunk != "[DONE]":
                    try:
                        last = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
        return last
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise MCPError(f"unparseable response: {raw[:300]}") from None


def _unwrap_content(content: list) -> object:
    """MCP wraps results as [{type:'text', text:'...'}]. Unwrap and re-parse."""
    texts = []
    for part in content or []:
        if isinstance(part, dict):
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif part.get("type") == "resource":
                texts.append(json.dumps(part.get("resource", {}), ensure_ascii=False))
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            return texts[0]
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except json.JSONDecodeError:
        return joined


def _flatten(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------
# harness adapter
# --------------------------------------------------------------------------


class LunitMCP:
    """Implements the MCPTools interface l2_harness expects.

    Caches the tool list and every result seen, so cite_uid resolution
    never costs an extra call.
    """

    def __init__(self, url: str = MCP_URL, key: str = API_KEY):
        self.rpc = StreamableHTTPClient(url, key)
        self._schemas: list[dict] | None = None
        self._cite_cache: dict[str, dict] = {}

    def schemas(self) -> list[dict]:
        if self._schemas is None:
            self._schemas = [
                {"name": t["name"],
                 "description": t.get("description", ""),
                 "parameters": t.get("inputSchema") or t.get("input_schema") or
                               {"type": "object", "properties": {}}}
                for t in self.rpc.list_tools()
            ]
        return self._schemas

    def call(self, name: str, arguments: dict):
        result = self.rpc.call_tool(name, arguments or {})
        self._cache(result)
        return result

    def safe_call(self, name: str, arguments: dict):
        try:
            return self.call(name, arguments)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)[:300]}

    def resolve_cite_uids(self, uids: list[str]) -> list[dict]:
        return [self._cite_cache[u] for u in uids if u in self._cite_cache]

    def _cache(self, obj) -> None:
        if isinstance(obj, dict):
            uid = obj.get("cite_uid")
            if uid:
                self._cite_cache[uid] = obj
            for v in obj.values():
                self._cache(v)
        elif isinstance(obj, list):
            for v in obj:
                self._cache(v)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _find_cite_uids(obj, out: list) -> None:
    if isinstance(obj, dict):
        if obj.get("cite_uid"):
            out.append(obj["cite_uid"])
        for v in obj.values():
            _find_cite_uids(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _find_cite_uids(v, out)


# Minimal probe arguments per tool. Adjust once you see the real schemas —
# `python3 lunit_mcp.py schema <tool>` prints them.
CITECHECK_PROBES = [
    ("index_get_relevant_nodes", {"corpus_tag": "guideline", "query": "hypertension"}),
    ("index_list_documents", {"corpus_tag": "guideline"}),
    ("index_keyword_search", {"corpus_tag": "guideline", "keywords": ["hypertension"]}),
    ("rag_get_all_data_sources", {}),
    ("rag_vector_query", {"data_source": "pubmed_abstracts", "query": "hypertension chronic kidney disease"}),
    ("kcd_search_codes", {"name": "type 2 diabetes"}),
    ("openapi_hira_get_drug_price", {"name": "pembrolizumab"}),
    ("openapi_mfds_check_drug_permission", {"name": "keytruda"}),
    ("adr_retrieve_drug_info", {"name": "metformin"}),
    ("hira_updates_search", {"query": "면역항암제"}),
    ("openapi_law_search", {"query": "의료법"}),
]


def cmd_list(client: LunitMCP) -> None:
    tools = client.schemas()
    print(f"{len(tools)} tools\n")
    for t in tools:
        desc = (t["description"] or "").replace("\n", " ")[:88]
        print(f"  {t['name']:38s} {desc}")


def cmd_schema(client: LunitMCP, name: str) -> None:
    for t in client.schemas():
        if t["name"] == name:
            print(json.dumps(t, indent=2, ensure_ascii=False))
            return
    print(f"no such tool: {name}", file=sys.stderr)
    sys.exit(1)


def cmd_call(client: LunitMCP, name: str, args_json: str) -> None:
    args = json.loads(args_json) if args_json else {}
    t0 = time.monotonic()
    result = client.call(name, args)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str)[:6000])
    uids: list = []
    _find_cite_uids(result, uids)
    print(f"\n  elapsed {time.monotonic() - t0:.1f}s   cite_uids: {uids or 'NONE'}")


def cmd_citecheck(client: LunitMCP) -> None:
    """Which tools emit cite_uid? This decides your retrieval architecture."""
    available = {t["name"] for t in client.schemas()}
    citable, non_citable, failed = [], [], []
    for name, args in CITECHECK_PROBES:
        if name not in available:
            continue
        try:
            result = client.call(name, args)
        except Exception as e:  # noqa: BLE001
            failed.append((name, str(e)[:70]))
            continue
        uids: list = []
        _find_cite_uids(result, uids)
        (citable if uids else non_citable).append(name)

    print("\nCITABLE (finalize_retrieval can carry these):")
    for n in citable:
        print(f"  + {n}")
    print("\nNOT CITABLE (values here are LOST unless your code rescues them):")
    for n in non_citable:
        print(f"  ! {n}")
    if failed:
        print("\nprobe failed (likely wrong argument names — check `schema <tool>`):")
        for n, e in failed:
            print(f"  ? {n}: {e}")
    print("\n-> Every tool in the second list must have its result carried to")
    print("   generation by your own assembly code, not by finalize_retrieval.")


def cmd_dump_assets(client: LunitMCP, outdir: str = "assets") -> None:
    """Cache what never changes, so eval-time tool calls aren't spent on it."""
    os.makedirs(outdir, exist_ok=True)
    jobs = [
        ("guideline_docs.txt", "index_list_documents", {"corpus_tag": "guideline"}),
        ("hira_docs.txt", "index_list_documents", {"corpus_tag": "hira"}),
        ("sql_schemas.txt", "rag_get_all_data_sources", {}),
    ]
    for filename, tool, args in jobs:
        try:
            result = client.call(tool, args)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {filename}: {e}", file=sys.stderr)
            continue
        path = os.path.join(outdir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False, indent=1, default=str))
        print(f"  wrote {path}  ({os.path.getsize(path)} bytes)")
    print(f"\n-> commit {outdir}/ and set L2_STATIC_DIR={outdir}")
    print("   l2_harness injects these into the retrieval prompt at startup.")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    try:
        client = LunitMCP()
        if cmd == "list":
            cmd_list(client)
        elif cmd == "schema":
            cmd_schema(client, sys.argv[2])
        elif cmd == "call":
            cmd_call(client, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "{}")
        elif cmd == "citecheck":
            cmd_citecheck(client)
        elif cmd == "dump-assets":
            cmd_dump_assets(client, sys.argv[2] if len(sys.argv) > 2 else "assets")
        else:
            print(f"unknown command: {cmd}", file=sys.stderr)
            sys.exit(1)
    except MCPError as e:
        print(f"\nMCP error: {e}\n", file=sys.stderr)
        print("  401/403 -> wrong key, or key is for the dashboard not the MCP server", file=sys.stderr)
        print("  timeout -> off-network, or the server is not open yet", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
