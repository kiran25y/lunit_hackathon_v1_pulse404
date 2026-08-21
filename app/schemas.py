from typing import Any, Literal
from pydantic import BaseModel, Field

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

class ModelTurn(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)

class CitableItem(BaseModel):
    cite_uid: str
    relevance_score: float = Field(ge=0, le=1)

class CitationSelection(BaseModel):
    status: Literal["sufficient", "partial", "no_evidence"]
    items: list[CitableItem] = Field(default_factory=list)
    note: str = ""

class Evidence(BaseModel):
    cite_uid: str
    source_type: str = "unknown"
    title: str = ""
    url: str = ""
    content: str = ""
    tool_name: str = ""

class PipelineResult(BaseModel):
    answer: str
    retrieval_status: str
    evidence: list[Evidence] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)

