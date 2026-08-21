import json
import re
import time
from pathlib import Path

from .conversation import build_case_state, self_contained_query
from .mcp_client import capture_citations
from .prompts import (
    FINALIZE_TOOL,
    GENERATION_PROMPT,
    RETRIEVAL_PROMPT,
    REVISION_PROMPT,
)
from .risk import assess_risk
from .schemas import (
    CitationSelection,
    Evidence,
    PipelineResult,
)


def _assistant(turn):
    """Convert the internal model response into a chat message."""

    message = {
        "role": "assistant",
        "content": turn.content,
    }

    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(
                        call.arguments,
                        ensure_ascii=False,
                    ),
                },
            }
            for call in turn.tool_calls
        ]

    return message


class HealthBenchHarness:
    def __init__(self, l2, mcp, settings):
        self.l2 = l2
        self.mcp = mcp
        self.s = settings

    async def _retrieve(self, query):
        """
        Run the separate L2 retrieval stage.

        L2 can call any configured MCP tool and must eventually call
        finalize_retrieval.
        """

        citation_store = {}
        retrieval_log = []

        messages = [
            {
                "role": "system",
                "content": RETRIEVAL_PROMPT,
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        try:
            available_tools = (
                await self.mcp.schemas()
            ) + [FINALIZE_TOOL]
        except Exception as exc:
            # Retrieval is an enhancement, not a reason to make the model
            # endpoint unavailable. This also keeps startup/smoke probes
            # useful when evaluator credentials are absent or misconfigured.
            retrieval_log.append(
                {
                    "event": "mcp_discovery_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

            return (
                CitationSelection(
                    status="no_evidence",
                    items=[],
                    note="The evidence service was unavailable.",
                ),
                [],
                retrieval_log,
            )

        finalization_requested = False

        for step in range(self.s.retrieval_max_calls):
            tools_for_turn = (
                [FINALIZE_TOOL]
                if finalization_requested
                else available_tools
            )

            turn = await self.l2.chat(
                messages,
                tools_for_turn,
            )

            messages.append(_assistant(turn))

            if not turn.tool_calls:
                retrieval_log.append(
                    {
                        "step": step + 1,
                        "event": "no_tool_call",
                        "content_preview": turn.content[:500],
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Do not answer the medical question. "
                            "Continue searching with the available tools, "
                            "or call finalize_retrieval now."
                        ),
                    }
                )
                continue

            for call in turn.tool_calls:
                if call.name == "finalize_retrieval":
                    selection = CitationSelection.model_validate(
                        call.arguments
                    )

                    requested_cite_uids = [
                        item.cite_uid
                        for item in selection.items
                    ]

                    # Accept only cite_uid values that were actually observed
                    # in MCP tool results.
                    selection.items = [
                        item
                        for item in selection.items
                        if item.cite_uid in citation_store
                    ][: self.s.max_selected_citations]

                    evidence = [
                        Evidence.model_validate(
                            citation_store[item.cite_uid]
                        )
                        for item in selection.items
                    ]

                    invalid_cite_uids = [
                        cite_uid
                        for cite_uid in requested_cite_uids
                        if cite_uid not in citation_store
                    ]

                    # A "sufficient" result without evidence is invalid.
                    if selection.status == "sufficient" and not evidence:
                        selection.status = "no_evidence"
                        selection.note = (
                            "Retrieval claimed sufficient evidence, but "
                            "no valid cite_uid items were selected."
                        )

                    # Partial evidence with no valid citations is also
                    # effectively no evidence.
                    if selection.status == "partial" and not evidence:
                        selection.status = "no_evidence"

                        if not selection.note:
                            selection.note = (
                                "No valid citable evidence was retrieved."
                            )

                    retrieval_log.append(
                        {
                            "step": step + 1,
                            "tool": "finalize_retrieval",
                            "requested_status": call.arguments.get(
                                "status"
                            ),
                            "validated_status": selection.status,
                            "requested_cite_uids": requested_cite_uids,
                            "selected_cite_uids": [
                                item.cite_uid
                                for item in selection.items
                            ],
                            "invalid_cite_uids": invalid_cite_uids,
                            "note": selection.note,
                        }
                    )

                    return selection, evidence, retrieval_log

                if call.name not in self.mcp.by_name:
                    retrieval_log.append(
                        {
                            "step": step + 1,
                            "tool": call.name,
                            "status": "unknown_tool",
                            "arguments": call.arguments,
                        }
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(
                                {
                                    "error": (
                                        f"Unknown MCP tool: {call.name}"
                                    )
                                }
                            ),
                        }
                    )
                    continue

                try:
                    result = await self.mcp.execute(
                        call.name,
                        call.arguments,
                    )

                    previous_cite_uids = set(citation_store)

                    capture_citations(
                        result,
                        call.name,
                        citation_store,
                    )

                    new_cite_uids = sorted(
                        set(citation_store) - previous_cite_uids
                    )

                    # A Lunit MCP result may place cite_uid and page text in
                    # different nested fields. Prefer clean page text over a
                    # duplicated serialization of the complete MCP response.
                    structured_result = (
                        result.get("structuredContent")
                        or result.get("structured_content")
                        or {}
                    )

                    pages = (
                        structured_result.get("pages", [])
                        if isinstance(structured_result, dict)
                        else []
                    )

                    page_text = "\n\n".join(
                        (
                            f"Page {page.get('page')}:\n"
                            f"{page.get('text', '')}"
                        )
                        for page in pages
                        if isinstance(page, dict)
                        and page.get("text")
                    )

                    if not page_text:
                        page_text = json.dumps(
                            structured_result or result,
                            ensure_ascii=False,
                        )

                    for cite_uid in new_cite_uids:
                        stored_evidence = citation_store[
                            cite_uid
                        ]

                        content = stored_evidence.get(
                            "content",
                            "",
                        )

                        if (
                            not isinstance(content, str)
                            or not content.strip()
                        ):
                            stored_evidence["content"] = (
                                page_text[:30000]
                            )

                    # Once direct citable content is available, give L2 only
                    # finalize_retrieval on its next turn. L2 still decides
                    # whether the evidence is sufficient or partial.
                    if new_cite_uids:
                        finalization_requested = True

                    retrieval_log.append(
                        {
                            "step": step + 1,
                            "tool": call.name,
                            "status": "succeeded",
                            "arguments": call.arguments,
                            "new_cite_uids": new_cite_uids,
                            "total_cite_uids": len(citation_store),
                        }
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(
                                result,
                                ensure_ascii=False,
                            )[:30000],
                        }
                    )

                    if new_cite_uids:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Direct citable content has been "
                                    "retrieved. Do not call additional "
                                    "evidence tools. Decide whether the "
                                    "evidence is sufficient or partial, "
                                    "then call finalize_retrieval using "
                                    "only valid cite_uid values already "
                                    "observed."
                                ),
                            }
                        )

                except Exception as exc:
                    retrieval_log.append(
                        {
                            "step": step + 1,
                            "tool": call.name,
                            "status": "failed",
                            "arguments": call.arguments,
                            "error": str(exc),
                        }
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(
                                {
                                    "error": (
                                        f"Tool execution failed: {exc}"
                                    )
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )

        # The retrieval call budget was exhausted without a valid
        # finalize_retrieval call.
        fallback_cite_uids = sorted(citation_store)[
            : self.s.max_selected_citations
        ]

        fallback_evidence = [
            Evidence.model_validate(citation_store[cite_uid])
            for cite_uid in fallback_cite_uids
        ]

        status = (
            "partial"
            if fallback_evidence
            else "no_evidence"
        )

        selection = CitationSelection(
            status=status,
            items=[],
            note=(
                "Retrieval tool-call budget was exhausted before "
                "a valid finalize_retrieval call."
            ),
        )

        retrieval_log.append(
            {
                "event": "retrieval_budget_exhausted",
                "validated_status": status,
                "fallback_cite_uids": fallback_cite_uids,
            }
        )

        return selection, fallback_evidence, retrieval_log

    async def answer(self, messages):
        """Run generation, optional retrieval, verification and revision."""

        started = time.time()

        case = build_case_state(messages)
        risk = assess_risk(case)

        evidence = []
        retrieval_status = (
            "not_requested"
            if self.s.enable_retrieval
            else "disabled"
        )
        retrieval_log = []

        generation_messages = [
            {
                "role": "system",
                "content": GENERATION_PROMPT,
            },
            {
                "role": "system",
                "content": json.dumps(
                    {
                        "case": case,
                        "risk": risk,
                    },
                    ensure_ascii=False,
                ),
            },
            *messages,
        ]

        retrieve_tool_schema = {
            "type": "function",
            "function": {
                "name": "retrieve_relevant_content",
                "description": (
                    "Retrieve authoritative medical evidence using "
                    "a single self-contained query."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "A self-contained medical evidence "
                                "retrieval query."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

        generation_tools = (
            [retrieve_tool_schema]
            if self.s.enable_retrieval
            else []
        )

        first_turn = await self.l2.chat(
            generation_messages,
            generation_tools,
        )

        generation_messages.append(_assistant(first_turn))

        retrieve_call = next(
            (
                call
                for call in first_turn.tool_calls
                if call.name == "retrieve_relevant_content"
            ),
            None,
        )

        final_turn = first_turn

        if retrieve_call:
            original_query = retrieve_call.arguments.get(
                "query",
                "",
            ).strip()

            if not original_query:
                original_query = case.get(
                    "current_request",
                    "",
                )

            retrieval_query = self_contained_query(
                original_query,
                case,
            )

            (
                selection,
                evidence,
                retrieval_log,
            ) = await self._retrieve(retrieval_query)

            retrieval_status = selection.status

            # Defensive consistency check.
            if retrieval_status == "sufficient" and not evidence:
                retrieval_status = "no_evidence"
                selection.status = "no_evidence"

                if not selection.note:
                    selection.note = (
                        "No valid evidence was available."
                    )

            evidence_payload = {
                "status": retrieval_status,
                "note": selection.note,
                "evidence": [
                    {
                        "citation": index + 1,
                        **item.model_dump(),
                    }
                    for index, item in enumerate(evidence)
                ],
            }

            generation_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": retrieve_call.id,
                    "content": json.dumps(
                        evidence_payload,
                        ensure_ascii=False,
                    ),
                }
            )

            # Generation receives no tools after retrieval. It must now
            # produce the final answer.
            final_turn = await self.l2.chat(
                generation_messages,
                [],
            )

        answer = final_turn.content.strip()

        issues = self._verify(
            answer,
            evidence,
            risk,
            retrieval_status,
        )

        if self.s.enable_revision and issues:
            revision_messages = [
                {
                    "role": "system",
                    "content": REVISION_PROMPT,
                },
                *messages,
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "draft": answer,
                            "retrieval_status": retrieval_status,
                            "evidence": [
                                {
                                    "citation": index + 1,
                                    **item.model_dump(),
                                }
                                for index, item in enumerate(evidence)
                            ],
                            "issues": issues,
                        },
                        ensure_ascii=False,
                    ),
                },
            ]

            revised_turn = await self.l2.chat(
                revision_messages,
                [],
            )

            revised_answer = revised_turn.content.strip()

            if revised_answer:
                answer = revised_answer

            # Report issues for the revised final answer, not the draft.
            issues = self._verify(
                answer,
                evidence,
                risk,
                retrieval_status,
            )

        trace = {
            "case": case,
            "risk": risk,
            "retrieval_status": retrieval_status,
            "evidence_count": len(evidence),
            "issues": issues,
            "retrieval": retrieval_log,
            "latency_seconds": round(
                time.time() - started,
                3,
            ),
        }

        return PipelineResult(
            answer=answer,
            retrieval_status=retrieval_status,
            evidence=evidence,
            trace=trace,
        )

    def _verify(
        self,
        answer,
        evidence,
        risk,
        retrieval_status,
    ):
        """Run deterministic checks before returning the answer."""

        issues = []

        if not answer:
            issues.append("Empty answer")

        valid_citation_numbers = set(
            range(1, len(evidence) + 1)
        )

        used_citation_numbers = {
            int(number)
            for number in re.findall(
                r"\[(\d+)\]",
                answer,
            )
        }

        invalid_citation_numbers = (
            used_citation_numbers
            - valid_citation_numbers
        )

        if invalid_citation_numbers:
            issues.append(
                "Invalid citation numbers: "
                f"{sorted(invalid_citation_numbers)}"
            )

        if evidence and not used_citation_numbers:
            issues.append(
                "Retrieved evidence was available but the answer "
                "did not cite it."
            )

        if (
            retrieval_status == "sufficient"
            and not evidence
        ):
            issues.append(
                "Retrieval status was sufficient without evidence."
            )

        if (
            risk["possible_emergency"]
            and not re.search(
                r"emergency|urgent|응급|119|\bER\b",
                answer[:300],
                re.IGNORECASE,
            )
        ):
            issues.append(
                "Urgent action missing from the opening."
            )

        if re.search(
            r"FAERS.*(causes|incidence|발생률|원인)",
            answer,
            re.IGNORECASE | re.DOTALL,
        ):
            issues.append(
                "Possible FAERS causality or incidence overclaim."
            )

        if re.search(
            r"\b(retrieval tools?|evidence service|available tools?|"
            r"internal (?:pipeline|processing)|system prompt)\b",
            answer,
            re.IGNORECASE,
        ):
            issues.append(
                "The answer exposed internal retrieval or processing details."
            )

        return issues

    def save_trace(self, request_id, result):
        """Save a local debugging trace."""

        trace_directory = Path(self.s.trace_dir)
        trace_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        trace_path = trace_directory / f"{request_id}.json"

        trace_path.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
