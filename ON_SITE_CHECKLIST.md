# On-Site Integration Checklist

## First 30 minutes

1. Obtain the exact L2 base URL, model ID, authentication method, request schema and tool-call response examples.
2. Obtain every MCP tool name, description, parameter schema, server URL and authentication rule.
3. Confirm whether MCP uses JSON-RPC `tools/call` or an organizer-provided Python client.
4. Confirm dashboard repository entry point, input schema, output schema, timeout, concurrency and dependency limits.
5. Ask whether multiple L2 generation calls per example are permitted and whether traces may be written locally.

## Adapter test

1. Set `L2_MODE=openai_compatible` and run `python -m app.main smoke`.
2. If the request fails, modify only `app/l2_client.py` using the official example.
3. Add one MCP tool to `config/mcp_tools.json` and test a query that must call it.
4. Inspect the trace and verify that its `cite_uid` appears in generation evidence.
5. Add remaining tools one at a time.

## Required cases before leaderboard use

- General medical question with no retrieval
- Current clinical-guideline question
- Korean drug label question
- HIRA reimbursement question
- FAERS question
- Multi-turn pronoun/reference question
- Clear emergency
- Conditional emergency
- Missing critical context
- Tool returns no evidence
- Tool timeout/error
- Invalid citation selection

## Final freeze

1. Run tests and submission check.
2. Run a representative private development slice.
3. Confirm no keys are staged.
4. Tag the exact tested commit.
5. Submit that commit, not a later untested edit.

