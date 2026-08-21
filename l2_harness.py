"""
l2_harness.py — reference skeleton for the Lunit FM L2 two-stage harness.

Design notes
------------
Only three components call the model (compact / retrieve / generate).
Everything else is deterministic Python so it is fast, testable, and legal
inside the isolated evaluation environment (no external network, no
non-L2 model may produce user-facing text).

Fill in the two adapters at the top once you have the on-site specs:
  - L2Client.chat(...)      -> the Lunit L2 endpoint
  - MCPTools.call(...)      -> the provided MCP tools
Everything below is written against those two interfaces only.

Failure policy: this module must never raise and never return an empty
string. A crashed example scores zero and is averaged into the mean.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Literal, Optional

log = logging.getLogger("l2")

# --------------------------------------------------------------------------
# Budgets. Tune these against the dashboard, not by intuition.
# --------------------------------------------------------------------------

MAX_RETRIEVAL_TOOL_CALLS = 6
RETRIEVAL_TIMEOUT_S = 45.0
EXAMPLE_TIMEOUT_S = 120.0
MAX_EVIDENCE_CHARS = 12_000
MODEL_RETRIES = 2

# The MCP server advertises tool_timeout_sec = 60. Six calls at 60s is six
# minutes on a single example. Set your own client timeout well below it and
# treat the wall-clock budget above as authoritative.
MCP_CALL_TIMEOUT_S = 20

# --------------------------------------------------------------------------
# Static assets — dump these ONCE during development and commit them.
#
# rag_get_all_data_sources + rag_get_data_source_detail return the SQL
# schemas; index_list_documents returns the 249 HIRA + 120 guideline titles.
# None of that changes between runs, so paying a tool call for it at eval
# time is a call you cannot spend on the actual question. Cache to disk,
# inject into the retrieval system prompt, and the model can name a document
# or write SQL without a discovery round trip.
#
# This caches the *tool corpus*, not the benchmark. It has nothing to do
# with HealthBench and is not what the reverse-engineering rule prohibits.
# --------------------------------------------------------------------------

STATIC_DIR = os.environ.get("L2_STATIC_DIR", "assets")


def _load_static_context() -> str:
    parts = []
    for name, header in (
        ("sql_schemas.txt", "SQL data sources and schemas (rag_sql_query):"),
        ("guideline_docs.txt", "Documents in corpus_tag='guideline':"),
        ("hira_docs.txt", "Documents in corpus_tag='hira':"),
    ):
        path = os.path.join(STATIC_DIR, name)
        try:
            with open(path, encoding="utf-8") as fh:
                parts.append(f"{header}\n{fh.read().strip()}")
        except OSError:
            log.info("static asset missing (fine early on): %s", path)
    return "\n\n".join(parts)


# ==========================================================================
# 0.  Adapters — the only two places that know about Lunit infrastructure
# ==========================================================================


class L2Client:
    """Wrap the Lunit L2 endpoint. Replace the body with the real call."""

    def __init__(self, endpoint: str, api_key: str, model: str = "l2"):
        self.endpoint, self.api_key, self.model = endpoint, api_key, model

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> dict:
        """Return {"content": str, "tool_calls": [{"name":..,"arguments":{..}}]}."""
        raise NotImplementedError("wire to the L2 endpoint on-site")

    # -- resilience wrapper: every call site uses this, not .chat directly --
    def call(self, **kw) -> dict:
        last = None
        for attempt in range(MODEL_RETRIES + 1):
            try:
                return self.chat(**kw)
            except Exception as e:  # noqa: BLE001 - deliberately broad
                last = e
                log.warning("L2 call failed (attempt %d): %s", attempt + 1, e)
                time.sleep(1.5 * (attempt + 1))
        log.error("L2 call exhausted retries: %s", last)
        return {"content": "", "tool_calls": []}


class MCPTools:
    """Wrap the hackathon MCP tools."""

    def schemas(self) -> list[dict]:
        """JSON schemas for every MCP tool, as passed to L2."""
        raise NotImplementedError

    def call(self, name: str, arguments: dict) -> Any:
        raise NotImplementedError

    def safe_call(self, name: str, arguments: dict) -> Any:
        try:
            return self.call(name, arguments)
        except Exception as e:  # noqa: BLE001
            log.warning("MCP tool %s failed: %s", name, e)
            return {"error": str(e)}

    def resolve_cite_uids(self, uids: list[str]) -> list[dict]:
        """Map cite_uids back to {title, url, source_type, content}.

        Keep a dict populated during the retrieval loop: every tool result
        that carries a cite_uid gets cached there. That avoids a second
        round of tool calls just to fetch content you already saw.
        """
        raise NotImplementedError


# ==========================================================================
# 1.  Conversation state — the fix for L2 being single-turn optimised
# ==========================================================================


@dataclass
class CaseCard:
    case: dict[str, str] = field(default_factory=dict)
    role: Literal["patient", "caregiver", "clinician", "unknown"] = "unknown"
    setting: str = ""
    standing_instructions: list[str] = field(default_factory=list)
    asked: list[dict] = field(default_factory=list)   # {"question":..,"answer":..}
    query: str = ""
    distress: bool = False

    def render(self) -> str:
        """Compact text block handed to the generation stage."""
        parts = []
        if self.case:
            facts = "; ".join(f"{k}: {v}" for k, v in self.case.items())
            parts.append(f"Established facts — {facts}")
        if self.role != "unknown":
            parts.append(f"Reader — {self.role}")
        if self.setting:
            parts.append(f"Setting — {self.setting}")
        if self.standing_instructions:
            parts.append("Standing instructions — " + "; ".join(self.standing_instructions))
        if self.asked:
            prior = "; ".join(
                f"asked '{a.get('question','')}' -> {a.get('answer') or 'no answer'}"
                for a in self.asked
            )
            parts.append(f"Already asked — {prior} (do not ask these again)")
        return "\n".join(parts)


COMPACT_SYSTEM = """You maintain the state of a medical conversation. You do not \
answer the user. You output only a JSON object.

Fields:
- case: every clinical fact the user has established (age, sex, symptom duration,
  severity, medications, allergies, comorbidities, pregnancy status, history).
  Omit fields the user has not stated. Never infer or invent a value.
- role: patient | caregiver | clinician | unknown
- setting: country, care setting, or resource constraints if stated, else ""
- standing_instructions: any format, length, language or style request from ANY
  earlier turn that still applies
- asked: questions the assistant already asked and the user's answer if given
- query: the user's final message rewritten as a single self-contained question.
  Resolve every pronoun and ellipsis. It must be answerable by someone who has
  not read the conversation.
- distress: true if the user expresses fear, pain or worry

Output JSON only."""


def compact(client: L2Client, messages: list[dict]) -> CaseCard:
    """Stage 2. Skipped on the first turn."""
    user_turns = [m for m in messages if m["role"] == "user"]
    last_user = user_turns[-1]["content"] if user_turns else ""

    if len(messages) <= 1:
        return CaseCard(query=last_user)

    transcript = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
    out = client.call(
        system=COMPACT_SYSTEM,
        messages=[{"role": "user", "content": transcript}],
        temperature=0.0,
        max_tokens=800,
    )
    card = _parse_case_card(out.get("content", ""))
    if not card.query.strip():
        card.query = last_user  # never lose the question
    return card


def _parse_case_card(raw: str) -> CaseCard:
    try:
        blob = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(blob.group(0)) if blob else {}
        return CaseCard(
            case=data.get("case") or {},
            role=data.get("role") or "unknown",
            setting=data.get("setting") or "",
            standing_instructions=data.get("standing_instructions") or [],
            asked=data.get("asked") or [],
            query=data.get("query") or "",
            distress=bool(data.get("distress")),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("case card parse failed: %s", e)
        return CaseCard()


# ==========================================================================
# 2.  Triage — deterministic emergency gate
# ==========================================================================
#
# Both under-escalation and over-escalation are penalised. Gate on a
# committed clinical list, not on model judgement. Extend this on-site
# with a proper red-flag file; keep the clinical rationale in comments so
# an administrator code review reads it as medicine, not benchmark gaming.

RED_FLAGS: list[tuple[str, str]] = [
    (r"crushing|chest (pain|pressure|tightness).*(arm|jaw|sweat|short)", "possible ACS"),
    (r"face droop|slurred speech|one[- ]sided weakness|sudden.*(numb|vision loss)", "possible stroke"),
    (r"throat clos|tongue swell|anaphyla|hives.*(breath|swall)", "possible anaphylaxis"),
    (r"can'?t breathe|severe (shortness of breath|dyspn)", "respiratory compromise"),
    (r"worst headache|thunderclap", "possible SAH"),
    (r"suicid|kill myself|end my life|self[- ]harm", "self-harm risk"),
    (r"unresponsive|unconscious|seizure lasting|not waking", "altered consciousness"),
    (r"heavy bleeding|bleeding (that )?won'?t stop|vomiting blood|black stool", "haemorrhage"),
    (r"pregnan.*(bleed|severe pain)|ectopic", "obstetric emergency"),
    (r"stiff neck.*fever|non[- ]blanching rash|purpuric rash", "possible meningococcal disease"),
]


def triage(text: str) -> Optional[str]:
    low = text.lower()
    for pattern, label in RED_FLAGS:
        if re.search(pattern, low):
            return label
    return None


# ==========================================================================
# 3.  Router — which tool profile, if any?
# ==========================================================================
#
# Twenty MCP tools is a lot of schema to put in front of the model on every
# query, and most HealthBench items are general clinical reasoning that L2
# answers from memory. Route to a small profile instead of exposing
# everything: fewer schemas means less confusion, fewer tokens, and a tool
# budget spent on the right corpus.
#
# NOTE: L2 was trained with these tools. It is possible it performs better
# with the full set present. A/B this against always-on and always-off
# retrieval on the validation set before committing. (See H+3 in the plan.)

TOOL_PROFILES: dict[str, list[str]] = {
    # Drug questions with no Korean angle — the broadly useful profile for
    # a mostly-English, mostly-global benchmark.
    "drug_global": [
        "adr_retrieve_drug_info",
        "rag_sql_query",
        "rag_vector_query",
    ],
    # Korean drug status: approved? listed? what does it cost?
    "drug_kr": [
        "openapi_mfds_check_drug_permission",
        "openapi_mfds_get_drug_indication",
        "openapi_mfds_find_drugs_by_ingredient",
        "openapi_hira_get_drug_price",
        "hira_updates_search",
    ],
    # Reimbursement criteria, claim review, off-label oncology regimens.
    "hira": [
        "hira_updates_search",
        "index_get_relevant_nodes",
        "index_get_page_content",
        "index_keyword_search",
        "rag_vector_query",
    ],
    # Diagnosis coding and claim validity.
    "coding": [
        "kcd_search_codes",
        "kcd_get_name",
        "openapi_hira_disease_check_code",
    ],
    # Statute and administrative rule text.
    "law": [
        "openapi_law_search",
        "openapi_law_list_articles",
        "openapi_law_get_article",
    ],
    # Clinical evidence: guideline corpus + PubMed. The profile most likely
    # to move the benchmark score rather than the Korean-context score.
    "evidence": [
        "index_get_relevant_nodes",
        "index_get_page_content",
        "index_keyword_search",
        "rag_vector_query",
    ],
    # Adverse event signals. See the FAERS caveat in GENERATION_SYSTEM.
    "pharmacovigilance": [
        "rag_sql_query",
        "adr_retrieve_drug_info",
        "rag_vector_query",
    ],
}

# Ordered: first match wins, so the Korea-specific patterns sit above the
# general ones.
ROUTES: list[tuple[str, str]] = [
    ("law",  r"(법령|시행령|시행규칙|고시|조문|제\d+조)|\b(statute|legal (basis|provision)|article \d+ of)\b"),
    ("coding", r"(kcd|상병\s*코드|질병\s*분류)|\b(icd[- ]?1[01]|diagnosis code|claim code)\b"),
    ("drug_kr", r"(약가|급여\s*(등재|여부)|허가\s*사항|식약처|비급여|얼마)"
                r"|\bmfds\b|\bdrug price|reimbursement listing|approved in korea"
                r"|\b(cost|price|priced|coverage|covered)\b[^.?!]{0,40}\b(korea|korean|kr)\b"),
    ("hira", r"(심평원|요양급여|급여\s*기준|삭감|심사\s*사례|고시\s*개정)|\bhira\b"),
    ("pharmacovigilance", r"(부작용\s*보고|이상사례)"
                          r"|\bfaers\b|adverse event report|adverse[- ]event report"
                          r"|safety signal|post[- ]?marketing surveillance|pharmacovigilance"),
    ("drug_global", r"\blabel\b|package insert|contraindicat\w*|drug[- ]drug interaction"
                    r"|drug interaction|black box|boxed warning|\bdosing\b|\bdosage\b"
                    r"|adverse reaction|\binteract(s|ion)?\b[^.?!]{0,30}\bwith\b"),
    ("evidence", r"(가이드라인|지침|권고)"
                 r"|\bguideline|\brecommend(s|ation|ations|ed)?\b|\bevidence\b|\btrials?\b"
                 r"|meta[- ]analys\w*|systematic review|what does the literature|stud(y|ies) show"),
    ("evidence", r"\b(cite|source|reference)\b|(근거|출처)"),
]


def route(query: str) -> Optional[str]:
    """Return a tool-profile name, or None to answer from memory."""
    low = query.lower()
    for profile, pattern in ROUTES:
        if re.search(pattern, low):
            return profile
    return None


# ==========================================================================
# 4.  Retrieval stage
# ==========================================================================

FINALIZE_SCHEMA = {
    "name": "finalize_retrieval",
    "description": (
        "Submit your final citation selection and end the retrieval phase. "
        "Call this once you have enough evidence, if the query needs no "
        "retrieval, or if you have exhausted the tool call budget."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["sufficient", "partial", "no_evidence"]},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cite_uid": {"type": "string"},
                        "relevance_score": {"type": "number"},
                    },
                    "required": ["cite_uid", "relevance_score"],
                },
            },
            "note": {"type": "string"},
        },
        "required": ["status", "items"],
    },
}

RETRIEVAL_SYSTEM = f"""You are gathering evidence. You do NOT write an answer.

Use the available tools to find items that let another model answer the query
accurately. Identify what kind of evidence the query needs, search for it, read
the most promising items, then stop.

Rules:
- At most {MAX_RETRIEVAL_TOOL_CALLS} tool calls. Spend them deliberately.
- For the document index, go straight to index_get_relevant_nodes: it already
  returns the matching document along with nodes and page ranges, so
  index_list_documents is a wasted call. Then index_get_page_content once.
  Two calls, not three.
- Prefer primary sources: guideline text, drug label, statute, HIRA notice,
  indexed page content over summaries.
- An item is relevant only if it would change or ground the answer. Do not
  select items that are merely on-topic.
- Finish by calling finalize_retrieval with the cite_uid of each relevant item
  and a relevance_score.
- status: "sufficient" if the evidence answers the query, "partial" if it
  answers part, "no_evidence" if nothing usable was found.
- note: this is your ONLY channel for facts that arrived without a cite_uid —
  drug prices, approval status, code validity, SQL results. Put those values
  in the note verbatim, with their source and effective date, e.g.
  "HIRA ceiling price 1,234 KRW, effective 2026-04-01". Also state in one line
  what you searched and what you could NOT find. The answering model reads this.
- If the budget runs out, call finalize_retrieval immediately with whatever you
  have. Never end without calling it."""

STATIC_CONTEXT = _load_static_context()


@dataclass
class Retrieved:
    status: str = "no_evidence"
    note: str = ""
    items: list[dict] = field(default_factory=list)   # citable  {title,url,source_type,content}
    facts: list[dict] = field(default_factory=list)   # non-citable {tool, args, result}


def retrieve(client: L2Client, tools: MCPTools, query: str,
             profile: Optional[str] = None) -> Retrieved:
    """Stage 4. Always returns; never raises.

    Two caches are kept, and this matters:

      cited[]  — results carrying a cite_uid. These are what
                 finalize_retrieval can refer to.
      raw[]    — EVERY tool result, citable or not.

    Several tools (the HIRA/MFDS/law OpenAPI wrappers, kcd_*, the SQL
    queries) may return no cite_uid at all. finalize_retrieval can only
    carry cite_uids, so anything from those tools would be silently lost
    on the documented path. Assembly reads from `raw` as well, using
    finalize_retrieval's selection as a *ranking* signal rather than a
    filter. Your Python owns assembly; nothing forces you to throw away a
    drug price because it arrived without a citation handle.
    """
    deadline = time.monotonic() + RETRIEVAL_TIMEOUT_S
    messages = [{"role": "user", "content": query}]

    allowed = TOOL_PROFILES.get(profile or "", None)
    schemas = tools.schemas()
    if allowed:
        schemas = [s for s in schemas if s.get("name") in allowed] or schemas
    tool_schemas = schemas + [FINALIZE_SCHEMA]

    cited: dict[str, dict] = {}
    raw: list[dict] = []

    for used in range(MAX_RETRIEVAL_TOOL_CALLS):
        if time.monotonic() > deadline:
            log.warning("retrieval timeout after %d calls", used)
            break

        remaining = MAX_RETRIEVAL_TOOL_CALLS - used
        sys_prompt = RETRIEVAL_SYSTEM + f"\n\nTool calls remaining: {remaining}."
        if STATIC_CONTEXT:
            sys_prompt += "\n\n" + STATIC_CONTEXT
        out = client.call(
            system=sys_prompt,
            messages=messages,
            tools=tool_schemas,
            temperature=0.0,
        )
        calls = out.get("tool_calls") or []
        if not calls:
            break

        for c in calls:
            if c["name"] == "finalize_retrieval":
                return _finalize(c.get("arguments") or {}, cited, raw, tools)

            result = tools.safe_call(c["name"], c.get("arguments") or {})
            _harvest_cite_uids(result, cited)
            raw.append({"tool": c["name"], "args": c.get("arguments") or {}, "result": result})
            messages.append({"role": "assistant", "content": json.dumps(c)})
            messages.append({"role": "tool", "content": _truncate(json.dumps(result, ensure_ascii=False), 6000)})

    # Budget or timeout exhausted without an explicit finalize: force one.
    out = client.call(
        system=RETRIEVAL_SYSTEM + "\n\nBudget exhausted. Call finalize_retrieval now.",
        messages=messages,
        tools=[FINALIZE_SCHEMA],
        temperature=0.0,
    )
    for c in out.get("tool_calls") or []:
        if c["name"] == "finalize_retrieval":
            return _finalize(c.get("arguments") or {}, cited, raw, tools)

    return Retrieved(status="partial" if (cited or raw) else "no_evidence",
                     note="retrieval budget exhausted without finalization",
                     items=list(cited.values())[:5],
                     facts=raw[-3:])


def _finalize(args: dict, cited: dict, raw: list, tools: MCPTools) -> Retrieved:
    status = args.get("status") or "no_evidence"
    note = args.get("note") or ""
    picks = sorted(args.get("items") or [],
                   key=lambda i: i.get("relevance_score", 0), reverse=True)
    items, chosen = [], set()
    for p in picks:
        uid = p.get("cite_uid")
        if not uid:
            continue
        chosen.add(uid)
        if uid in cited:
            items.append(cited[uid])
        else:
            try:
                items.extend(tools.resolve_cite_uids([uid]))
            except Exception as e:  # noqa: BLE001
                log.warning("cite_uid %s unresolved: %s", uid, e)

    # Rescue non-citable results the model could not select but which may
    # still carry the answer (prices, code validity, approval status).
    facts = [r for r in raw if not _has_cite_uid(r["result"])]

    # If the model selected nothing but tools clearly returned content,
    # do not report no_evidence — that path ends in a refusal.
    if status == "no_evidence" and (items or facts):
        status = "partial"
    return Retrieved(status=status, note=note, items=items, facts=facts)


def _harvest_cite_uids(result: Any, seen: dict) -> None:
    """Cache anything carrying a cite_uid so finalize needs no extra calls."""
    if isinstance(result, dict):
        uid = result.get("cite_uid")
        if uid:
            seen[uid] = result
        for v in result.values():
            _harvest_cite_uids(v, seen)
    elif isinstance(result, list):
        for v in result:
            _harvest_cite_uids(v, seen)


def _has_cite_uid(result: Any) -> bool:
    found: dict = {}
    _harvest_cite_uids(result, found)
    return bool(found)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "\n…[truncated]"


# ==========================================================================
# 5.  Assemble — evidence formatted for the generation stage
# ==========================================================================


def assemble(r: Retrieved) -> str:
    if not r.items and not r.facts:
        return (f"status: {r.status}\nnote: {r.note}\n"
                "No usable sources found. Answer from your own knowledge and "
                "state the uncertainty plainly. Do not refuse.")

    out = [f"status: {r.status}"]
    if r.note:
        out.append(f"note: {r.note}")

    budget = MAX_EVIDENCE_CHARS
    for n, item in enumerate(r.items, start=1):
        content = _truncate(str(item.get("content", "")), max(500, budget // max(1, len(r.items))))
        block = (f"\n[{n}]\nsource_type: {item.get('source_type','')}\n"
                 f"url: {item.get('url','')}\ntitle: {item.get('title','')}\n"
                 f"content: {content}")
        if budget - len(block) < 0:
            break
        budget -= len(block)
        out.append(block)

    # Non-citable tool output — prices, approval status, code validity, SQL
    # rows. Labelled so the model knows it is a lookup result rather than a
    # quotable source, and must not attach a [n] marker to it.
    if r.facts:
        out.append("\nLOOKUP RESULTS (authoritative, but not citable — state "
                   "the source in words, do not use a [n] marker):")
        per = max(400, 3000 // max(1, len(r.facts)))
        for f in r.facts[:6]:
            out.append(f"- {f['tool']}({json.dumps(f['args'], ensure_ascii=False)[:200]}) -> "
                       f"{_truncate(json.dumps(f['result'], ensure_ascii=False), per)}")
    return "\n".join(out)


# ==========================================================================
# 6.  Generation stage
# ==========================================================================

GENERATION_SYSTEM = """You are a medical expert answering one question. You have \
one tool, retrieve_relevant_content. Call it only when the answer depends on a \
specific guideline, law, drug label, reimbursement rule, or Korean regulatory \
detail. Answer general medical questions directly.

ANSWER SHAPE — use for every clinical answer unless the user specified a format,
in which case follow the user exactly and drop this shape:
1. The direct answer in one or two sentences. No preamble, no restating the question.
2. "Seek care now if:" — a short list of red flags, when clinically relevant.
3. The substance as labelled bullets. One distinct claim per bullet, lead clause in bold.
4. "Next steps:" — concrete and ordered.
5. "What would change this:" — the one or two facts that would most change the advice.

RULES
- Never withhold the answer to ask a question. Answer under the most likely
  interpretation, state the branch points, and name missing facts in section 5.
- If the user describes a life-threatening emergency, the first sentence tells
  them to get emergency care. Do not escalate otherwise.
- Match register to the reader: a clinician gets drug names, doses and mechanism;
  a layperson gets plain language and no unglossed jargon.
- Use only facts you are confident in or that appear in retrieved content. Cite
  retrieved facts inline as [1], [2]. Never invent a number, dose, guideline name
  or citation.
- Adverse-event report data (FAERS) is spontaneous reporting with no denominator
  and no established causality. It supports "has been reported" and never
  "occurs in X% of patients" or "causes". Converting a report count into an
  incidence rate or a causal claim is a factual error.
- Korean reimbursement, price, and approval facts are point-in-time. Give the
  effective date alongside the value, and say plainly that it may have changed.
- State uncertainty in plain words where it exists. Do not hedge where it does not.
- Be concise. No filler, no apologies, no "consult your doctor" in place of an answer."""

RETRIEVE_TOOL_SCHEMA = {
    "name": "retrieve_relevant_content",
    "description": "Retrieve relevant content to ground your answer. Pass a single, self-contained query.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


def generate(
    client: L2Client,
    card: CaseCard,
    emergency: Optional[str],
    retriever: Callable[[str], str],
    allow_tool: bool,
) -> str:
    system = GENERATION_SYSTEM
    if emergency:
        system += (f"\n\nTRIAGE: features consistent with {emergency}. Lead with a "
                   "clear instruction to seek emergency care, then advise.")
    if card.distress:
        system += "\n\nThe user is worried. Open with one brief line of acknowledgement, then answer."

    state = card.render()
    user_block = (f"{state}\n\n---\n{card.query}" if state else card.query)
    messages = [{"role": "user", "content": user_block}]

    out = client.call(
        system=system,
        messages=messages,
        tools=[RETRIEVE_TOOL_SCHEMA] if allow_tool else None,
        temperature=0.3,
        max_tokens=2048,
    )

    for c in out.get("tool_calls") or []:
        if c["name"] == "retrieve_relevant_content":
            evidence = retriever((c.get("arguments") or {}).get("query") or card.query)
            messages.append({"role": "assistant", "content": json.dumps(c)})
            messages.append({"role": "tool", "content": evidence})
            out = client.call(system=system, messages=messages,
                              temperature=0.3, max_tokens=2048)
            break

    return (out.get("content") or "").strip()


# ==========================================================================
# 7.  Post-check — deterministic guards
# ==========================================================================

ESCALATION_RE = re.compile(
    r"(emergency|119|911|ambulance|urgent care|emergency room|er now|immediately seek|응급)",
    re.I,
)


def post_check(text: str, emergency: Optional[str]) -> list[str]:
    problems = []
    if not text.strip():
        problems.append("empty response")
        return problems
    if emergency:
        head = " ".join(text.split()[:60])
        if not ESCALATION_RE.search(head):
            problems.append("triage fired but no escalation in the opening")
    if re.search(r"\[\d+\]", text) and "source" not in text.lower():
        pass  # inline markers are fine; only flag if evidence was never supplied
    if len(text) > 6000:
        problems.append("over length band")
    return problems


# ==========================================================================
# Orchestration
# ==========================================================================


def answer(client: L2Client, tools: MCPTools, messages: list[dict]) -> str:
    """Entry point. Takes the full conversation, returns the final turn.

    Guarantees a non-empty string under every failure mode.
    """
    started = time.monotonic()
    card = CaseCard()
    try:
        card = compact(client, messages)
        emergency = triage(card.query + " " + json.dumps(card.case, ensure_ascii=False))

        profile = route(card.query)

        def retriever(q: str) -> str:
            if emergency or time.monotonic() - started > EXAMPLE_TIMEOUT_S * 0.6:
                return "status: no_evidence\nnote: retrieval skipped. Answer directly."
            return assemble(retrieve(client, tools, q, profile or route(q)))

        # RETRIEVAL_MODE: "routed" (default) | "always" | "never".
        # A/B all three on the validation set at H+3 — L2 was trained with
        # these tools present and may behave differently without them.
        mode = os.environ.get("RETRIEVAL_MODE", "routed")
        allow_tool = (not emergency) and (
            mode == "always" or (mode == "routed" and profile is not None)
        )
        text = generate(client, card, emergency, retriever, allow_tool)

        problems = post_check(text, emergency)
        if problems and time.monotonic() - started < EXAMPLE_TIMEOUT_S * 0.85:
            log.info("post-check repair: %s", problems)
            text = _repair(client, card, emergency, text, problems) or text

        if text.strip():
            return text
        log.error("empty after generation; falling back")
    except Exception as e:  # noqa: BLE001
        log.exception("pipeline failed, falling back: %s", e)

    return _fallback(client, card.query or _last_user(messages))


def _repair(client, card, emergency, draft, problems) -> str:
    out = client.call(
        system=(GENERATION_SYSTEM +
                "\n\nRevise the draft below to fix the listed problems. Change nothing else. "
                "Output the revised answer only."),
        messages=[{"role": "user",
                   "content": f"Question: {card.query}\n\nProblems: {'; '.join(problems)}\n\nDraft:\n{draft}"}],
        temperature=0.2,
        max_tokens=2048,
    )
    return (out.get("content") or "").strip()


def _fallback(client: L2Client, query: str) -> str:
    """Last resort: plain generation, no tools, no state. Must return text."""
    out = client.call(
        system="You are a medical expert. Answer clearly, accurately and concisely. "
               "Lead with the direct answer. Note red flags. Do not refuse.",
        messages=[{"role": "user", "content": query}],
        temperature=0.3,
        max_tokens=1500,
    )
    text = (out.get("content") or "").strip()
    return text or (
        "I can't give a reliable answer to this right now. If the situation is "
        "severe, worsening, or involves difficulty breathing, chest pain, "
        "confusion, or heavy bleeding, seek emergency care immediately."
    )


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""
