from time import perf_counter
from typing import Any, Callable

from rag import get_embedding_provider
from tools import ToolResult, llm_match_analysis_tool, rag_retrieve_tool
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
    }


def run_resume_agent_workflow(
    resume_text: str,
    job_description: str,
    top_k: int = 5,
    use_rag: bool = True,
    source_name: str = "resume_text",
    section_filter: str | None = None,
    llm_callable: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Run the fixed tool chain and return analysis plus a lightweight trace."""
    workflow_started = perf_counter()
    configured_provider = get_embedding_provider() if use_rag else None
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
    )
    retrieved_chunks: list[dict[str, Any]] = []
    retrieval_data: dict[str, Any] = {}

    if use_rag:
        retrieve_result, retrieve_step = _timed_tool_step(
            step_name="retrieve_resume_context",
            input_summary={
                "job_description_preview": summarize_text(job_description),
                "resume_length": len(resume_text or ""),
                "top_k": top_k,
                "section_filter": section_filter,
            },
            tool_call=lambda: rag_retrieve_tool(
                resume_text=resume_text,
                job_description=job_description,
                top_k=top_k,
                source_name=source_name,
                section_filter=section_filter,
            ),
            output_builder=lambda result: {
                **summarize_chunks(result.data.get("chunks", [])),
                "embedding_provider": result.data.get("embedding_provider"),
            },
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
        trace.used_fallback = bool(
            configured_provider
            and trace.embedding_provider
            and configured_provider != trace.embedding_provider
        )
    else:
        timestamp = now_iso()
        trace.steps.append(
            TraceStep(
                step_name="retrieve_resume_context",
                tool_name="rag_retrieve_tool",
                success=True,
                message="RAG retrieval skipped because use_rag=False.",
                input_summary={"use_rag": False, "top_k": top_k},
                output_summary={"skipped": True, "retrieved_chunk_count": 0},
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
        },
        tool_call=lambda: llm_match_analysis_tool(
            resume_text=resume_text,
            job_description=job_description,
            retrieved_chunks=retrieved_chunks,
            use_rag=use_rag,
            llm_callable=llm_callable,
        ),
        output_builder=lambda result: {
            "analysis_length": len(result.data.get("full_report", "")),
            "has_result": bool(result.data.get("full_report")),
            "retrieved_chunk_count": result.data.get("retrieved_chunk_count", 0),
        },
    )
    trace.steps.append(llm_step)

    if not llm_result.success:
        message = f"Agent Workflow stopped at LLM analysis: {llm_result.error}"
        return _base_result(
            trace, workflow_started, False, _error_analysis(message), retrieved_chunks,
            top_k, section_filter, retrieval_data, error=message,
        )

    return _base_result(
        trace, workflow_started, True, llm_result.data, retrieved_chunks,
        top_k, section_filter, retrieval_data,
    )
