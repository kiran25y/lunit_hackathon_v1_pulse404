import json, uuid
import certifi
import httpx
from .schemas import ModelTurn, ToolCall

class L2Client:
    def __init__(self, settings): self.s = settings

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ModelTurn:
        if self.s.l2_mode == "mock": return self._mock(messages, tools or [])
        if self.s.l2_mode != "openai_compatible":
            raise ValueError(f"Unsupported L2_MODE={self.s.l2_mode}; adapt app/l2_client.py")
        headers = {"Content-Type": "application/json"}
        if self.s.l2_api_key: headers["Authorization"] = f"Bearer {self.s.l2_api_key}"
        payload = {"model": self.s.l2_model, "messages": messages, "temperature": 0}
        if tools: payload.update(tools=tools, tool_choice="auto")
        async with httpx.AsyncClient(
            timeout=self.s.timeout,
            verify=certifi.where(),
        ) as client:
            r = await client.post(f"{self.s.l2_base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
            r.raise_for_status(); msg = r.json()["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls", []):
            args = c["function"].get("arguments", {})
            if isinstance(args, str): args = json.loads(args or "{}")
            calls.append(ToolCall(id=c.get("id", str(uuid.uuid4())), name=c["function"]["name"], arguments=args))
        return ModelTurn(content=msg.get("content") or "", tool_calls=calls)

    def _mock(self, messages, tools) -> ModelTurn:
        names = [t.get("function", {}).get("name") for t in tools]
        if "finalize_retrieval" in names:
            return ModelTurn(tool_calls=[ToolCall(id="mock-final", name="finalize_retrieval", arguments={"status":"no_evidence","items":[],"note":"Mock mode"})])
        if "retrieve_relevant_content" in names and not any(m.get("role") == "tool" for m in messages):
            return ModelTurn(tool_calls=[ToolCall(id="mock-retrieve", name="retrieve_relevant_content", arguments={"query":"self-contained mock medical query"})])
        return ModelTurn(content="Mock L2 response. Configure the on-site Lunit endpoint for medical answers.")
