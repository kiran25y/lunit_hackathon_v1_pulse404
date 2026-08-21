import os
import time
import json
import asyncio
import requests
import httpx

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


app = FastAPI()


# ============================================================
# CONFIG
# ============================================================

API_KEY = os.environ["LUNIT_FM_API_KEY"]

MODEL_URL = os.environ.get(
    "LUNIT_FM_API_URL",
    "https://model.hackathon.lunit.io",
)

MODEL_NAME = os.environ.get(
    "LUNIT_FM_MODEL",
    "Lunit/L2-preview",
)

MCP_URL = os.environ.get(
    "LUNIT_MCP_URL",
    "https://mcp.hackathon.lunit.io/mcp",
)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


# ============================================================
# API SCHEMAS
# ============================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = None


# ============================================================
# PROMPTS
# ============================================================

ROUTER_PROMPT = """
You are a routing system for a Korean medical assistant.

Given the latest user message and conversation context,
choose exactly ONE primary route from:

general
pubmed
drug_label
guideline
hira
law
mfds

Definitions:

general:
ordinary medical knowledge that does not require checking a specific
study, official drug label, Korean guideline, reimbursement rule,
Korean law, or Korean drug approval.

pubmed:
specific research paper, study, trial, numerical research result,
scientific claim, publication, AI model performance, cohort study,
meta-analysis, or "is there really a study saying X?"

drug_label:
drug adverse effects, warnings, contraindications, interactions,
official label safety information, or whether a symptom is a known
drug adverse reaction.

guideline:
clinical practice guideline recommendations, diagnostic algorithms,
treatment pathways, screening recommendations, or professional
guidance.

hira:
Korean reimbursement, 급여, 비급여, 인정 기준, 심사 기준,
HIRA notices, coverage decisions, or reimbursement cases.

law:
Korean laws, legal qualifications, statutory requirements,
regulations, legal duties, or exact legal provisions.

mfds:
whether a drug is approved in Korea, approved indications,
dosage/administration, contraindications, or official MFDS product
approval information.

Return ONLY valid JSON in this exact form:

{
  "route": "general|pubmed|drug_label|guideline|hira|law|mfds",
  "query": "short search query",
  "drug_name": "English generic or brand name if relevant, else null"
}

Do not include markdown.
""".strip()


FINAL_SYSTEM_PROMPT = """
You are a careful Korean medical assistant.

Use the full conversation context.

If external evidence is provided:
- Ground your answer in that evidence.
- Do not invent facts not supported by the evidence.
- Distinguish what the source directly shows from your interpretation.
- If evidence is incomplete, say so clearly.
- Preserve important numerical values exactly when relevant.
- Mention source names such as PubMed, DailyMed, HIRA, MFDS,
  clinical guideline, or Korean law when helpful.
- Do not fabricate citations or URLs.

For medical advice:
- Explain uncertainty.
- Avoid making a definitive diagnosis without appropriate clinical data.
- Recommend professional evaluation when clinically appropriate.

Answer primarily in Korean unless the user asks otherwise.
Be concise but useful.
""".strip()


# ============================================================
# LUNIT FM CALL
# ============================================================

def call_lunit(messages, temperature=0.2):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }

    for attempt in range(3):
        response = requests.post(
            f"{MODEL_URL}/v1/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=120,
        )

        if response.status_code == 502:
            time.sleep(3)
            continue

        if not response.ok:
            raise RuntimeError(
                f"Lunit model error: "
                f"{response.status_code} {response.text}"
            )

        return response.json()

    raise RuntimeError(
        "Lunit model failed after repeated 502 errors."
    )


# ============================================================
# ROUTER
# ============================================================


def choose_route(messages):
    latest = messages[-1]["content"]
    lower = latest.lower()

    # --------------------------------------------------------
    # 1. DETERMINISTIC ROUTING FOR HIGH-CONFIDENCE CASES
    # --------------------------------------------------------

    # HIRA / Korean reimbursement
    hira_keywords = [
        "hira",
        "급여",
        "비급여",
        "급여기준",
        "보험 적용",
        "보험적용",
        "심사기준",
        "건강보험",
        "수가",
    ]

    if any(k in lower for k in hira_keywords):
        return {
            "route": "hira",
            "query": latest,
            "drug_name": None,
        }

    # Clinical guideline
    guideline_keywords = [
        "임상진료지침",
        "진료지침",
        "가이드라인",
        "guideline",
        "권고안",
        "권고하나요",
        "권고하나요?",
        "권고",
    ]

    if any(k in lower for k in guideline_keywords):
        return {
            "route": "guideline",
            "query": latest,
            "drug_name": None,
        }

    # Korean law / regulation
    law_keywords = [
        "법",
        "법률",
        "법적",
        "법령",
        "시행령",
        "시행규칙",
        "자격 요건",
        "자격요건",
        "법적 요건",
    ]

    if any(k in lower for k in law_keywords):
        return {
            "route": "law",
            "query": latest,
            "drug_name": None,
        }

    # Korea drug approval / MFDS
    mfds_keywords = [
        "식약처",
        "mfds",
        "한국에서 승인",
        "국내 승인",
        "허가됐",
        "허가되었",
        "허가 여부",
        "승인 여부",
        "적응증",
    ]

    if any(k in lower for k in mfds_keywords):
        return {
            "route": "mfds",
            "query": latest,
            "drug_name": None,
        }

    # Drug label / safety
    drug_keywords = [
        "부작용",
        "이상반응",
        "adverse",
        "side effect",
        "금기",
        "contraindication",
        "약물 상호작용",
        "상호작용",
        "경고",
        "warning",
    ]

    if any(k in lower for k in drug_keywords):
        # Ask L2 only to extract the English drug name
        extraction_prompt = [
            {
                "role": "system",
                "content": (
                    "Extract the English generic or brand drug name "
                    "from the user's message. Return only the drug name. "
                    "If no drug is present, return NONE."
                ),
            },
            {
                "role": "user",
                "content": latest,
            },
        ]

        try:
            result = call_lunit(
                extraction_prompt,
                temperature=0.0,
            )

            drug_name = (
                result["choices"][0]["message"]["content"]
                .strip()
                .replace('"', "")
            )

            if drug_name.upper() == "NONE":
                drug_name = None

        except Exception:
            drug_name = None

        return {
            "route": "drug_label",
            "query": latest,
            "drug_name": drug_name,
        }

    # Specific paper / study / numerical research claim
    pubmed_keywords = [
        "연구",
        "논문",
        "study",
        "paper",
        "pubmed",
        "pmid",
        "임상시험",
        "코호트",
        "메타분석",
        "정확도",
        "auc",
        "민감도",
        "특이도",
        "카파",
        "kappa",
    ]

    if any(k in lower for k in pubmed_keywords):
        return {
            "route": "pubmed",
            "query": latest,
            "drug_name": None,
        }

    # --------------------------------------------------------
    # 2. L2 ROUTER FOR AMBIGUOUS QUESTIONS
    # --------------------------------------------------------

    recent_messages = messages[-6:]

    router_messages = [
        {
            "role": "system",
            "content": ROUTER_PROMPT,
        },
        *recent_messages,
    ]

    try:
        result = call_lunit(
            router_messages,
            temperature=0.0,
        )

        raw = (
            result["choices"][0]["message"]["content"]
            .strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        data = json.loads(raw)

    except Exception:
        return {
            "route": "general",
            "query": latest,
            "drug_name": None,
        }

    route = data.get("route", "general")

    allowed = {
        "general",
        "pubmed",
        "drug_label",
        "guideline",
        "hira",
        "law",
        "mfds",
    }

    if route not in allowed:
        route = "general"

    return {
        "route": route,
        "query": data.get("query", latest),
        "drug_name": data.get("drug_name"),
    }


# ============================================================
# MCP CALL
# ============================================================

async def call_mcp_tool(tool_name, arguments):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }

    async with httpx.AsyncClient(
        headers=headers,
        follow_redirects=True,
        timeout=120.0,
    ) as http_client:

        async with streamable_http_client(
            MCP_URL,
            http_client=http_client,
        ) as (read_stream, write_stream):

            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:

                await session.initialize()

                result = await session.call_tool(
                    tool_name,
                    arguments=arguments,
                )

                if hasattr(result, "model_dump"):
                    return result.model_dump(
                        mode="json",
                        by_alias=True,
                    )

                return str(result)


def run_mcp_tool(tool_name, arguments):
    return asyncio.run(
        call_mcp_tool(
            tool_name,
            arguments,
        )
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_evidence(route_info):
    route = route_info["route"]
    query = route_info["query"]
    drug_name = route_info.get("drug_name")

    if route == "general":
        return None

    try:
        if route == "pubmed":
            return {
                "source": "PubMed",
                "tool": "rag_vector_query",
                "result": run_mcp_tool(
                    "rag_vector_query",
                    {
                        "query": query,
                        "collection_name": "pubmed_abstracts",
                        "top_k": 5,
                    },
                ),
            }

        if route == "drug_label":
            if not drug_name:
                return None

            return {
                "source": "DailyMed",
                "tool": "adr_retrieve_drug_info",
                "result": run_mcp_tool(
                    "adr_retrieve_drug_info",
                    {
                        "drug_name": drug_name,
                    },
                ),
            }

        if route == "guideline":
            return {
                "source": "Clinical guideline",
                "tool": "index_get_relevant_nodes",
                "result": run_mcp_tool(
                    "index_get_relevant_nodes",
                    {
                        "corpus_tag": "guideline",
                        "query": query,
                        "k": 5,
                    },
                ),
            }

        if route == "hira":
            return {
                "source": "HIRA",
                "tool": "hira_updates_search",
                "result": run_mcp_tool(
                    "hira_updates_search",
                    {
                        "query": query,
                        "current_only": True,
                        "limit": 5,
                        "search_mode": "both",
                        "document_type": "all",
                        "source_type": "all",
                    },
                ),
            }

        if route == "law":
            return {
                "source": "Korean Law Information Center",
                "tool": "openapi_law_search",
                "result": run_mcp_tool(
                    "openapi_law_search",
                    {
                        "query": query,
                    },
                ),
            }

        if route == "mfds":
            if not drug_name:
                return None

            return {
                "source": "MFDS",
                "tool": "openapi_mfds_get_drug_indication",
                "result": run_mcp_tool(
                    "openapi_mfds_get_drug_indication",
                    {
                        "drug_name": drug_name,
                    },
                ),
            }

    except Exception as error:
        return {
            "source": "MCP",
            "tool": route,
            "error": str(error),
        }

    return None


# ============================================================
# FINAL ANSWER
# ============================================================

def generate_final_answer(messages, route_info, evidence):
    system_content = FINAL_SYSTEM_PROMPT

    if evidence:
        evidence_text = json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2,
        )

        # Prevent huge MCP payloads from exploding prompt size
        evidence_text = evidence_text[:16000]

        system_content += (
            "\n\n"
            "Retrieved evidence follows.\n"
            "Use it carefully and do not claim more than it supports.\n\n"
            f"{evidence_text}"
        )

    final_messages = [
        {
            "role": "system",
            "content": system_content,
        },
        *messages,
    ]

    result = call_lunit(
        final_messages,
        temperature=0.2,
    )

    return result


# ============================================================
# OPENAI-COMPATIBLE ENDPOINTS
# ============================================================

@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "lunit-hackathon-agent",
                "object": "model",
                "owned_by": "team",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    messages = [
        {
            "role": m.role,
            "content": m.content,
        }
        for m in request.messages
    ]

    if not messages:
        raise HTTPException(
            status_code=400,
            detail="messages cannot be empty",
        )

    try:
        route_info = choose_route(messages)

        print("\n" + "=" * 60)
        print("ROUTE:", route_info)
        print("=" * 60)

        evidence = retrieve_evidence(
            route_info
        )

        if evidence:
            print("MCP SOURCE:", evidence.get("source"))
            print("MCP TOOL:", evidence.get("tool"))
            if evidence.get("error"):
                print("MCP ERROR:", evidence.get("error"))
        else:
            print("MCP: not used")

        result = generate_final_answer(
            messages,
            route_info,
            evidence,
        )

        content = (
            result["choices"][0]["message"]["content"]
        )

        return {
            "id": result.get(
                "id",
                "chatcmpl-lunit-agent",
            ),
            "object": "chat.completion",
            "created": result.get("created"),
            "model": "lunit-hackathon-agent",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": result.get(
                "usage",
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "lunit-hackathon-agent",
    }
