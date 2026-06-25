from time import perf_counter
from typing import Any, Callable

from rag import get_embedding_provider
from llm_provider import get_llm_provider_from_env
from tools import ToolResult, llm_match_analysis_tool, rag_retrieve_tool, review_report_tool
from trace_utils import (
    TraceStep,
    WorkflowTrace,
    create_run_id,
    now_iso,
    save_trace_json,
    summarize_chunks,
    summarize_text,
    trace_to_dict,
)


def _step_from_trace(step: TraceStep) -> dict[str, Any]:
    """Keep the original workflow_steps view compatible with Streamlit."""
    return {
        "step_name": step.step_name,
        "tool_name": step.tool_name,
        "success": step.success,
        "message": step.message,
        "data_summary": step.output_summary,
        "error": step.error,
    }


def _error_analysis(message: str) -> dict[str, str]:
    return {
        "job_analysis": message,
        "resume_analysis": message,
        "match_analysis": message,
        "suggestions": message,
        "full_report": message,
    }


def _rerank_summary(result: ToolResult) -> dict[str, Any]:
    chunks = result.data.get("chunks", [])
    rerank_scores = [
        float(chunk["rerank_score"])
        for chunk in chunks
        if chunk.get("rerank_score") is not None
    ]
    keyword_hits = sorted(
        {
            keyword
            for chunk in chunks
            for keyword in chunk.get("keyword_hits", [])
        }
    )
    return {
        **summarize_chunks(chunks),
        "embedding_provider": result.data.get("embedding_provider"),
        "used_rerank": result.data.get("used_rerank", False),
        "rerank_method": result.data.get("rerank_method"),
        "reranked_chunk_count": len(chunks),
        "rerank_keyword_hits": keyword_hits,
        "rerank_score": max(rerank_scores) if rerank_scores else None,
    }


def _timed_tool_step(
    step_name: str,
    input_summary: dict[str, Any],
    tool_call: Callable[[], ToolResult],
    output_builder: Callable[[ToolResult], dict[str, Any]],
) -> tuple[ToolResult, TraceStep]:
    start_time = now_iso()
    started = perf_counter()
    result = tool_call()
    duration_ms = round((perf_counter() - started) * 1000, 2)

    return result, TraceStep(
        step_name=step_name,
        tool_name=result.tool_name,
        success=result.success,
        message=result.message,
        input_summary=input_summary,
        output_summary=output_builder(result),
        error=result.error,
        start_time=start_time,
        end_time=now_iso(),
        duration_ms=duration_ms,
    )


def _finish_trace(
    trace: WorkflowTrace,
    workflow_started: float,
    success: bool,
    error: str | None = None,
) -> tuple[dict[str, Any], str | None, str | None]:
    trace.end_time = now_iso()
    trace.duration_ms = round((perf_counter() - workflow_started) * 1000, 2)
    trace.final_status = "success" if success else "failed"
    trace.error = error

    trace_path = None
    trace_save_error = None
    try:
        trace_path = save_trace_json(trace)
    except Exception as exc:
        trace_save_error = str(exc)

    return trace_to_dict(trace), trace_path, trace_save_error


def _base_result(
    trace: WorkflowTrace,
    workflow_started: float,
    success: bool,
    analysis: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    top_k: int,
    section_filter: str | None,
    retrieval_data: dict[str, Any],
    error: str | None = None,
    review_result: dict[str, Any] | None = None,
    query_refinement_used: bool = False,
    retrieval_attempts: int = 1,
) -> dict[str, Any]:
    trace_dict, trace_path, trace_save_error = _finish_trace(
        trace,
        workflow_started,
        success=success,
        error=error,
    )
    return {
        **analysis,
        "success": success,
        "analysis": analysis.get("full_report", ""),
        "retrieved_chunks": retrieved_chunks,
        "workflow_steps": [_step_from_trace(step) for step in trace.steps],
        "trace": trace_dict,
        "trace_path": trace_path,
        "trace_save_error": trace_save_error,
        "error": error,
        "rag_sources": retrieved_chunks,
        "retrieved_context": retrieval_data.get("retrieved_context", ""),
        "retrieved_chunk_count": len(retrieved_chunks),
        "embedding_provider": retrieval_data.get("embedding_provider"),
        "rag_top_k": top_k,
        "rag_total_chunks": retrieval_data.get("rag_total_chunks"),
        "rag_section_filter": retrieval_data.get("rag_section_filter", section_filter),
        "rag_available_filtered_chunks": retrieval_data.get("rag_available_filtered_chunks"),
        "used_rerank": retrieval_data.get("used_rerank", False),
        "rerank_method": retrieval_data.get("rerank_method"),
        "llm_provider": trace.llm_provider,
        "llm_model": trace.llm_model,
        "use_mock_llm": trace.use_mock_llm,
        "fallback_to_mock": trace.fallback_to_mock,
        "fallback_used": trace.fallback_used,
        "original_provider": trace.original_provider,
        "provider_error": trace.provider_error,
        "review_result": review_result or {},
        "query_refinement_used": query_refinement_used,
        "retrieval_attempts": retrieval_attempts,
    }


def evaluate_retrieval_quality(retrieved_chunks: list[dict[str, Any]]) -> tuple[str, str]:
    """Simple heuristic to judge whether the first-pass retrieval is acceptable.

    Returns (quality, reason) where quality is "low" or "ok".
    """
    if not retrieved_chunks:
        return "low", "RAG 检索未返回任何片段。"

    has_keywords = any(chunk.get("keyword_hits") for chunk in retrieved_chunks)
    top_score = max(
        (float(chunk.get("rerank_score", 0) or 0) for chunk in retrieved_chunks),
        default=0,
    )

    if not has_keywords and top_score < 0.5:
        return "low", "top chunks 缺少关键词命中且 rerank 分数较低。"

    return "ok", "检索质量可接受。"


def run_resume_agent_workflow(
    resume_text: str,
    job_description: str,
    top_k: int = 5,
    use_rag: bool = True,
    source_name: str = "resume_text",
    section_filter: str | None = None,
    use_rerank: bool = False,
    llm_callable: Callable[[str], str] | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    use_mock_llm: bool = False,
    fallback_to_mock: bool = True,
) -> dict[str, Any]:
    """Run the fixed tool chain and return analysis plus a lightweight trace."""
    workflow_started = perf_counter()
    configured_provider = get_embedding_provider() if use_rag else None
    resolved_llm_provider = (
        "mock" if use_mock_llm or llm_callable is not None else (llm_provider or get_llm_provider_from_env())
    )
    effective_use_mock = bool(
        use_mock_llm or llm_callable is not None or resolved_llm_provider == "mock"
    )
    trace = WorkflowTrace(
        run_id=create_run_id(),
        mode="agent_workflow_rag" if use_rag else "agent_workflow_llm",
        start_time=now_iso(),
        end_time=None,
        duration_ms=None,
        resume_length=len(resume_text or ""),
        job_description_length=len(job_description or ""),
        top_k=top_k,
        embedding_provider=None,
        used_rag=use_rag,
        used_fallback=None,
        used_rerank=bool(use_rag and use_rerank),
        rerank_method="rule_based" if use_rag and use_rerank else None,
        llm_provider=resolved_llm_provider,
        llm_model=llm_model,
        use_mock_llm=effective_use_mock,
        fallback_to_mock=fallback_to_mock,
        review_passed=None,
        query_refinement_used=False,
        retrieval_attempts=1,
    )
    retrieved_chunks: list[dict[str, Any]] = []
    retrieval_data: dict[str, Any] = {}
    query_refinement_used = False
    retrieval_attempts = 1

    if use_rag:
        retrieve_result, retrieve_step = _timed_tool_step(
            step_name="retrieve_resume_context",
            input_summary={
                "job_description_preview": summarize_text(job_description),
                "resume_length": len(resume_text or ""),
                "top_k": top_k,
                "section_filter": section_filter,
                "use_rerank": use_rerank,
            },
            tool_call=lambda: rag_retrieve_tool(
                resume_text=resume_text,
                job_description=job_description,
                top_k=top_k,
                source_name=source_name,
                section_filter=section_filter,
                use_rerank=use_rerank,
            ),
            output_builder=_rerank_summary,
        )
        trace.steps.append(retrieve_step)

        if not retrieve_result.success:
            message = f"Agent Workflow stopped at RAG retrieval: {retrieve_result.error}"
            return _base_result(
                trace, workflow_started, False, _error_analysis(message), [], top_k,
                section_filter, retrieval_data, error=message,
            )

        retrieval_data = retrieve_result.data
        retrieved_chunks = retrieval_data.get("chunks", [])
        trace.embedding_provider = retrieval_data.get("embedding_provider")
        trace.used_rerank = retrieval_data.get("used_rerank", False)
        trace.rerank_method = retrieval_data.get("rerank_method")
        trace.used_fallback = bool(
            configured_provider
            and trace.embedding_provider
            and configured_provider != trace.embedding_provider
        )

        # --- bounded query refinement (max 1 retry) ---
        quality, reason = evaluate_retrieval_quality(retrieved_chunks)
        trace.steps[-1].output_summary["retrieval_quality"] = quality
        if quality == "low":
            refined_query = " ".join(job_description.split()[:80]) if job_description else job_description
            refine_result, refine_step = _timed_tool_step(
                step_name="retrieve_resume_context_retry",
                input_summary={
                    "refinement_reason": reason,
                    "original_top_k": top_k,
                    "refined_query_preview": summarize_text(refined_query),
                },
                tool_call=lambda: rag_retrieve_tool(
                    resume_text=resume_text,
                    job_description=refined_query,
                    top_k=top_k,
                    source_name=source_name,
                    section_filter=section_filter,
                    use_rerank=use_rerank,
                ),
                output_builder=_rerank_summary,
            )
            trace.steps.append(refine_step)
            query_refinement_used = True
            retrieval_attempts = 2
            if refine_result.success:
                refined_chunks = refine_result.data.get("chunks", [])
                if refined_chunks:
                    retrieved_chunks = refined_chunks
                    retrieval_data = refine_result.data
    else:
        timestamp = now_iso()
        trace.steps.append(
            TraceStep(
                step_name="retrieve_resume_context",
                tool_name="rag_retrieve_tool",
                success=True,
                message="RAG retrieval skipped because use_rag=False.",
                input_summary={"use_rag": False, "top_k": top_k, "use_rerank": False},
                output_summary={
                    "skipped": True,
                    "retrieved_chunk_count": 0,
                    "used_rerank": False,
                },
                start_time=timestamp,
                end_time=timestamp,
                duration_ms=0.0,
            )
        )

    llm_result, llm_step = _timed_tool_step(
        step_name="generate_match_analysis",
        input_summary={
            "resume_length": len(resume_text or ""),
            "job_description_preview": summarize_text(job_description),
            "retrieved_chunk_count": len(retrieved_chunks),
            "use_rag": use_rag,
            "use_rerank": bool(use_rag and use_rerank),
            "llm_provider": resolved_llm_provider,
            "llm_model": llm_model,
            "use_mock_llm": effective_use_mock,
            "fallback_to_mock": fallback_to_mock,
        },
        tool_call=lambda: llm_match_analysis_tool(
            resume_text=resume_text,
            job_description=job_description,
            retrieved_chunks=retrieved_chunks,
            use_rag=use_rag,
            llm_callable=llm_callable,
            llm_provider=llm_provider,
            llm_model=llm_model,
            use_mock_llm=effective_use_mock,
            fallback_to_mock=fallback_to_mock,
        ),
        output_builder=lambda result: {
            "analysis_length": len(result.data.get("full_report", "")),
            "has_result": bool(result.data.get("full_report")),
            "retrieved_chunk_count": result.data.get("retrieved_chunk_count", 0),
            "llm_provider": result.data.get("llm_provider"),
            "llm_model": result.data.get("llm_model"),
            "use_mock_llm": result.data.get("use_mock_llm", False),
            "fallback_to_mock": result.data.get("fallback_to_mock", True),
            "fallback_used": result.data.get("fallback_used", False),
            "original_provider": result.data.get("original_provider"),
            "provider_error": result.data.get("provider_error"),
        },
    )
    trace.steps.append(llm_step)
    trace.llm_provider = llm_result.data.get("llm_provider", trace.llm_provider)
    trace.llm_model = llm_result.data.get("llm_model", trace.llm_model)
    trace.use_mock_llm = llm_result.data.get("use_mock_llm", trace.use_mock_llm)
    trace.fallback_to_mock = llm_result.data.get("fallback_to_mock", trace.fallback_to_mock)
    trace.fallback_used = llm_result.data.get("fallback_used", False)
    trace.original_provider = llm_result.data.get("original_provider")
    trace.provider_error = llm_result.data.get("provider_error")

    if not llm_result.success:
        message = f"Agent Workflow stopped at LLM analysis: {llm_result.error}"
        trace.query_refinement_used = query_refinement_used
        trace.retrieval_attempts = retrieval_attempts
        return _base_result(
            trace, workflow_started, False, _error_analysis(message), retrieved_chunks,
            top_k, section_filter, retrieval_data, error=message,
            query_refinement_used=query_refinement_used,
            retrieval_attempts=retrieval_attempts,
        )

    # --- reviewer step ---
    analysis_text = llm_result.data.get("full_report", "")
    review_result, review_step = _timed_tool_step(
        step_name="review_match_analysis",
        input_summary={
            "analysis_length": len(analysis_text),
            "retrieved_chunk_count": len(retrieved_chunks),
        },
        tool_call=lambda: review_report_tool(
            job_description=job_description,
            retrieved_chunks=retrieved_chunks,
            analysis_text=analysis_text,
        ),
        output_builder=lambda result: {
            "review_passed": result.data.get("review_passed"),
            "missing_points_count": len(result.data.get("missing_points", [])),
            "risk_notes_count": len(result.data.get("risk_notes", [])),
            "review_summary": result.data.get("review_summary", ""),
        },
    )
    trace.steps.append(review_step)
    trace.review_passed = review_result.data.get("review_passed")
    trace.query_refinement_used = query_refinement_used
    trace.retrieval_attempts = retrieval_attempts

    return _base_result(
        trace, workflow_started, True, llm_result.data, retrieved_chunks,
        top_k, section_filter, retrieval_data,
        review_result=review_result.data,
        query_refinement_used=query_refinement_used,
        retrieval_attempts=retrieval_attempts,
    )
