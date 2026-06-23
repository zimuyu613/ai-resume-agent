from typing import Any, Callable

from tools import ToolResult, llm_match_analysis_tool, rag_retrieve_tool


def _summarize_data(data: dict[str, Any]) -> dict[str, Any]:
    """Keep workflow trace compact enough for Streamlit and interviews."""
    summary: dict[str, Any] = {}

    for key in [
        "retrieved_chunk_count",
        "rag_top_k",
        "rag_total_chunks",
        "embedding_provider",
        "use_rag",
    ]:
        if key in data:
            summary[key] = data[key]

    if "chunks" in data:
        summary["chunk_count"] = len(data.get("chunks") or [])

    if "full_report" in data:
        summary["analysis_char_count"] = len(data.get("full_report") or "")

    return summary


def _step_from_tool(step_name: str, result: ToolResult) -> dict[str, Any]:
    return {
        "step_name": step_name,
        "tool_name": result.tool_name,
        "success": result.success,
        "message": result.message,
        "data_summary": _summarize_data(result.data),
        "error": result.error,
    }


def _error_analysis(message: str) -> dict[str, str]:
    return {
        "job_analysis": message,
        "resume_analysis": message,
        "match_analysis": message,
        "suggestions": message,
        "full_report": message,
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
    """
    Fixed tool-calling workflow for the MVP.

    This is intentionally simple: retrieve optional RAG context, call the LLM
    analysis tool, then return both the final report and a readable trace.
    """
    workflow_steps: list[dict[str, Any]] = []
    retrieved_chunks: list[dict[str, Any]] = []
    retrieval_data: dict[str, Any] = {}

    if use_rag:
        retrieve_result = rag_retrieve_tool(
            resume_text=resume_text,
            job_description=job_description,
            top_k=top_k,
            source_name=source_name,
            section_filter=section_filter,
        )
        workflow_steps.append(_step_from_tool("retrieve_resume_context", retrieve_result))

        if not retrieve_result.success:
            message = f"Agent Workflow stopped at RAG retrieval: {retrieve_result.error}"
            return {
                **_error_analysis(message),
                "error": message,
                "workflow_steps": workflow_steps,
                "rag_sources": [],
                "retrieved_chunk_count": 0,
                "rag_top_k": top_k,
                "rag_section_filter": section_filter,
            }

        retrieval_data = retrieve_result.data
        retrieved_chunks = retrieval_data.get("chunks", [])

    llm_result = llm_match_analysis_tool(
        resume_text=resume_text,
        job_description=job_description,
        retrieved_chunks=retrieved_chunks,
        use_rag=use_rag,
        llm_callable=llm_callable,
    )
    workflow_steps.append(_step_from_tool("generate_match_analysis", llm_result))

    if not llm_result.success:
        message = f"Agent Workflow stopped at LLM analysis: {llm_result.error}"
        return {
            **_error_analysis(message),
            "error": message,
            "workflow_steps": workflow_steps,
            "rag_sources": retrieved_chunks,
            "retrieved_chunk_count": len(retrieved_chunks),
            "rag_top_k": top_k,
            "rag_section_filter": section_filter,
        }

    analysis = llm_result.data
    return {
        "job_analysis": analysis.get("job_analysis", ""),
        "resume_analysis": analysis.get("resume_analysis", ""),
        "match_analysis": analysis.get("match_analysis", ""),
        "suggestions": analysis.get("suggestions", ""),
        "full_report": analysis.get("full_report", ""),
        "workflow_steps": workflow_steps,
        "rag_sources": retrieved_chunks,
        "retrieved_context": retrieval_data.get("retrieved_context", ""),
        "retrieved_chunk_count": len(retrieved_chunks),
        "embedding_provider": retrieval_data.get("embedding_provider"),
        "rag_top_k": top_k,
        "rag_total_chunks": retrieval_data.get("rag_total_chunks"),
        "rag_section_filter": retrieval_data.get("rag_section_filter", section_filter),
        "rag_available_filtered_chunks": retrieval_data.get("rag_available_filtered_chunks"),
    }
