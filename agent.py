from prompts import (
    COMPREHENSIVE_ANALYSIS_PROMPT,
    RAG_ANALYSIS_PROMPT,
)
from rag import retrieve_relevant_chunks_with_sources
from rerank_utils import rerank_chunks
from llm_provider import LLMResult, generate_with_llm


def call_llm_result(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    use_mock: bool = False,
    timeout: int = 60,
    fallback_to_mock: bool = True,
) -> LLMResult:
    """Call the unified provider layer while prompt construction stays in agent.py."""
    return generate_with_llm(prompt, provider, model, use_mock, timeout, fallback_to_mock)


def call_llm(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    use_mock: bool = False,
    timeout: int = 60,
    fallback_to_mock: bool = True,
) -> str:
    """Backward-compatible text-only wrapper around the unified provider layer."""
    result = call_llm_result(prompt, provider, model, use_mock, timeout, fallback_to_mock)
    return result.text if result.success else f"错误：{result.error}"


def extract_section(text: str, title: str, next_title: str | None = None) -> str:
    """
    从完整报告中提取指定标题下的内容。
    如果提取失败，则返回完整报告，避免页面空白。
    """
    start_marker = f"## {title}"

    if start_marker not in text:
        return text

    start_index = text.find(start_marker)

    if next_title:
        next_marker = f"## {next_title}"
        end_index = text.find(next_marker, start_index + len(start_marker))

        if end_index != -1:
            return text[start_index:end_index].strip()

    return text[start_index:].strip()


def run_agent_workflow(
    job_description: str,
    resume_text: str,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    use_mock_llm: bool = False,
    fallback_to_mock: bool = True,
) -> dict:
    """
    单次综合调用版 Agent Workflow：
    通过一次模型调用完成四个分析模块，减少免费 API 高负载时的失败概率。

    逻辑上仍然保留四个步骤：
    1. 岗位要求分析
    2. 个人能力分析
    3. 匹配度分析
    4. 简历优化建议
    """

    job_description = job_description[:4000]
    resume_text = resume_text[:4000]
    
    prompt = COMPREHENSIVE_ANALYSIS_PROMPT.format(
        job_description=job_description,
        resume_text=resume_text,
    )

    llm_result = call_llm_result(
        prompt,
        provider=llm_provider,
        model=llm_model,
        use_mock=use_mock_llm,
        fallback_to_mock=fallback_to_mock,
    )
    full_report = llm_result.text if llm_result.success else f"错误：{llm_result.error}"

    job_analysis = extract_section(
        full_report,
        "岗位要求分析",
        "个人能力分析",
    )

    resume_analysis = extract_section(
        full_report,
        "个人能力分析",
        "匹配度分析",
    )

    match_analysis = extract_section(
        full_report,
        "匹配度分析",
        "简历优化建议",
    )

    suggestions = extract_section(
        full_report,
        "简历优化建议",
        None,
    )

    return {
        "job_analysis": job_analysis,
        "resume_analysis": resume_analysis,
        "match_analysis": match_analysis,
        "suggestions": suggestions,
        "full_report": full_report,
        "llm_provider": llm_result.provider,
        "llm_model": llm_result.model,
        "fallback_to_mock": fallback_to_mock,
        "fallback_used": llm_result.fallback_used,
        "original_provider": llm_result.original_provider,
        "provider_error": llm_result.error,
        "error": None if llm_result.success else llm_result.error,
    }


def _build_error_result(message: str, top_k: int | None = None, section_filter: str | None = None) -> dict:
    """把错误信息包装成页面可展示的四个 Tab，避免 Streamlit 页面崩溃。"""
    error_text = f"RAG 检索增强分析暂时无法完成，可能是 Gemini API 免费额度或请求频率限制。请稍后重试，或关闭 RAG 模式使用普通分析。\n\n错误详情：{message}"
    return {
        "job_analysis": error_text,
        "resume_analysis": error_text,
        "match_analysis": error_text,
        "suggestions": error_text,
        "full_report": error_text,
        "error": error_text,
        "rag_sources": [],
        "retrieved_chunk_count": 0,
        "rag_top_k": top_k,
        "rag_section_filter": section_filter,
    }


def run_rag_workflow(
    job_description: str,
    resume_text: str,
    source_name: str = "简历文本",
    top_k: int = 3,
    section_filter: str | None = None,
    use_rerank: bool = False,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    use_mock_llm: bool = False,
    fallback_to_mock: bool = True,
) -> dict:
    """
    RAG 增强版分析流程：
    1. 根据岗位描述召回最相关的简历片段
    2. 将岗位描述和片段填入 RAG Prompt
    3. 调用 Gemini 生成完整报告
    4. 复用 extract_section 拆分为四个页面 Tab
    """
    try:
        candidate_k = min(max(top_k * 2, top_k), 12) if use_rerank else top_k
        retrieval_result = retrieve_relevant_chunks_with_sources(
            job_description=job_description,
            resume_text=resume_text,
            source_name=source_name,
            top_k=candidate_k,
            section_filter=section_filter,
        )
    except Exception as e:
        return _build_error_result(str(e), top_k=top_k, section_filter=section_filter)

    sources = retrieval_result.get("sources", [])
    if use_rerank:
        sources = rerank_chunks(sources, job_description=job_description, top_k=top_k)
    retrieved_context = "\n\n".join(
        f"[相关片段 {index} | section={source.get('section', 'unknown')}]\n{source.get('text', '')}"
        for index, source in enumerate(sources, start=1)
    )

    if not retrieved_context.strip():
        if section_filter:
            return _build_error_result(
                "当前检索范围没有召回到简历片段，请切换为“全部”或选择其他简历模块后重试。",
                top_k=top_k,
                section_filter=section_filter,
            )

        return _build_error_result(
            "没有召回到与岗位描述相关的简历片段，请检查简历文本是否足够完整。",
            top_k=top_k,
            section_filter=section_filter,
        )

    prompt = RAG_ANALYSIS_PROMPT.format(
        job_description=job_description[:4000],
        retrieved_context=retrieved_context,
    )

    llm_result = call_llm_result(
        prompt,
        provider=llm_provider,
        model=llm_model,
        use_mock=use_mock_llm,
        fallback_to_mock=fallback_to_mock,
    )
    full_report = llm_result.text if llm_result.success else f"错误：{llm_result.error}"

    job_analysis = extract_section(
        full_report,
        "岗位要求分析",
        "个人能力分析",
    )

    resume_analysis = extract_section(
        full_report,
        "个人能力分析",
        "匹配度分析",
    )

    match_analysis = extract_section(
        full_report,
        "匹配度分析",
        "简历优化建议",
    )

    suggestions = extract_section(
        full_report,
        "简历优化建议",
        None,
    )

    return {
        "job_analysis": job_analysis,
        "resume_analysis": resume_analysis,
        "match_analysis": match_analysis,
        "suggestions": suggestions,
        "full_report": full_report,
        "llm_provider": llm_result.provider,
        "llm_model": llm_result.model,
        "fallback_to_mock": fallback_to_mock,
        "fallback_used": llm_result.fallback_used,
        "original_provider": llm_result.original_provider,
        "provider_error": llm_result.error,
        "error": None if llm_result.success else llm_result.error,
        "retrieved_context": retrieved_context,
        "retrieved_chunk_count": len(sources),
        "rag_sources": sources,
        "embedding_provider": retrieval_result.get("embedding_provider"),
        "rag_top_k": top_k,
        "rag_total_chunks": retrieval_result.get("total_chunks"),
        "rag_section_filter": retrieval_result.get("section_filter"),
        "rag_available_filtered_chunks": retrieval_result.get("available_filtered_chunks"),
        "used_rerank": use_rerank,
        "rerank_method": "rule_based" if use_rerank else None,
    }
