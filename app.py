import json
import os
from html import escape
from io import BytesIO
from pathlib import Path

import streamlit as st
from docx import Document
from pypdf import PdfReader

from api_client import (
    call_agent_workflow_api,
    call_rag_retrieve_api,
    check_api_health,
    check_llm_health_api,
)
from agent_workflow import run_resume_agent_workflow
from agent import extract_section, run_agent_workflow, run_rag_workflow
from llm_provider import check_llm_provider_health, get_llm_provider_from_env
from rag import get_embedding_provider
from tools import llm_match_analysis_tool


BASE_DIR = Path(__file__).parent
SAMPLES_DIR = BASE_DIR / "samples"
SAMPLE_RESUME_PATH = SAMPLES_DIR / "sample_resume.txt"
SAMPLE_JOB_PATH = SAMPLES_DIR / "sample_job_description.txt"

SECTION_OPTIONS = {
    "全部": None,
    "基本信息": "basic_info",
    "项目经历": "project_experience",
    "技能栈": "skills",
    "实习经历": "internship_experience",
    "教育背景": "education",
    "获奖竞赛": "awards",
    "自我评价": "self_evaluation",
}

SECTION_LABELS = {value: label for label, value in SECTION_OPTIONS.items()}
SECTION_LABELS[None] = "全部"

LOCAL_BACKEND_MODE = "本地函数模式"
API_BACKEND_MODE = "FastAPI 接口模式"
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
APP_VERSION_LABEL = "v2.5 Lightweight Agent Harness"
LLM_PROVIDER_OPTIONS = ["gemini", "deepseek", "openai_compatible", "mock"]
LLM_PROVIDER_LABELS = {
    "gemini": "Gemini",
    "deepseek": "DeepSeek",
    "openai_compatible": "OpenAI Compatible",
    "mock": "Mock",
}
DEEPSEEK_MODEL_OPTIONS = ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "custom"]


GLOBAL_CSS = """
<style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .hero-panel {
        padding: 2rem 2.2rem;
        border: 1px solid #e6e9ef;
        border-radius: 14px;
        background: linear-gradient(135deg, #f8fbff 0%, #ffffff 48%, #f6f8fb 100%);
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 2.7rem;
        font-weight: 760;
        line-height: 1.05;
        margin-bottom: 0.3rem;
        color: #172033;
    }
    .hero-subtitle {
        font-size: 1.35rem;
        font-weight: 620;
        color: #344054;
        margin-bottom: 0.85rem;
    }
    .hero-desc {
        font-size: 1.02rem;
        color: #475467;
        max-width: 860px;
        line-height: 1.65;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.9rem 0 1.1rem;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.28rem 0.62rem;
        border: 1px solid #d7e3f4;
        border-radius: 999px;
        background: #f4f8ff;
        color: #24558f;
        font-size: 0.82rem;
        font-weight: 650;
        white-space: nowrap;
    }
    .section-title {
        margin-top: 1.3rem;
        margin-bottom: 0.8rem;
        padding-left: 0.75rem;
        border-left: 4px solid #2f80ed;
        font-size: 1.2rem;
        font-weight: 730;
        color: #172033;
    }
    .app-card {
        min-height: 134px;
        padding: 1rem 1.05rem;
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        background: #ffffff;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.055);
        margin-bottom: 0.85rem;
    }
    .app-card h4 {
        margin: 0 0 0.45rem;
        color: #172033;
        font-size: 1rem;
    }
    .app-card p {
        margin: 0;
        color: #5d6678;
        line-height: 1.55;
        font-size: 0.92rem;
    }
    .soft-card {
        padding: 1rem 1.05rem;
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        background: #fbfcfe;
        margin-bottom: 0.8rem;
    }
    .status-card {
        padding: 0.9rem 1rem;
        border-radius: 12px;
        border: 1px solid #e6e9ef;
        background: #ffffff;
        margin-bottom: 0.75rem;
    }
    .status-card.success {
        border-color: #a7e3c1;
        background: #f1fbf5;
    }
    .status-card.warning {
        border-color: #ffd89a;
        background: #fff8eb;
    }
    .status-card.info {
        border-color: #b9d7ff;
        background: #f3f8ff;
    }
    .status-card.danger {
        border-color: #ffc0c0;
        background: #fff3f3;
    }
    .status-title {
        font-weight: 730;
        margin-bottom: 0.25rem;
        color: #172033;
    }
    .status-text {
        color: #4b5565;
        line-height: 1.55;
    }
    .chunk-card {
        padding: 1rem;
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        background: #ffffff;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.045);
        margin-bottom: 0.85rem;
    }
    .chunk-meta {
        color: #667085;
        font-size: 0.86rem;
        line-height: 1.65;
        margin-top: 0.45rem;
    }
    .chunk-preview {
        color: #344054;
        font-size: 0.92rem;
        line-height: 1.62;
        margin-top: 0.55rem;
        white-space: pre-wrap;
    }
    .sidebar-footer {
        margin-top: 0.75rem;
        padding: 0.8rem;
        border-radius: 10px;
        background: #f6f8fb;
        border: 1px solid #e6e9ef;
        color: #475467;
        font-size: 0.82rem;
        line-height: 1.55;
    }
</style>
"""


def _analysis_error_result(message: str) -> dict:
    return {
        "success": False,
        "error": message,
        "job_analysis": message,
        "resume_analysis": message,
        "match_analysis": message,
        "suggestions": message,
        "full_report": message,
        "analysis": message,
        "rag_sources": [],
        "retrieved_chunks": [],
        "workflow_steps": [],
        "trace": {},
    }


def render_badges(labels: list[str]) -> None:
    badge_html = "".join(f'<span class="badge-pill">{escape(str(label))}</span>' for label in labels)
    st.markdown(f'<div class="badge-row">{badge_html}</div>', unsafe_allow_html=True)


def render_section_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)


def render_feature_card(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="app-card">
            <h4>{escape(title)}</h4>
            <p>{escape(text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(title: str, text: str, status: str = "info") -> None:
    st.markdown(
        f"""
        <div class="status-card {status}">
            <div class="status-title">{escape(title)}</div>
            <div class="status-text">{escape(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_bool(value) -> str:
    if value is True:
        return "是"
    if value is False:
        return "否"
    return "-"


def _split_analysis_report(full_report: str) -> dict:
    return {
        "job_analysis": extract_section(full_report, "岗位要求分析", "个人能力分析"),
        "resume_analysis": extract_section(full_report, "个人能力分析", "匹配度分析"),
        "match_analysis": extract_section(full_report, "匹配度分析", "简历优化建议"),
        "suggestions": extract_section(full_report, "简历优化建议", None),
        "full_report": full_report,
        "analysis": full_report,
    }


def _agent_api_result(api_result: dict, top_k: int, use_rag: bool) -> dict:
    if not api_result.get("success"):
        return _analysis_error_result(api_result.get("error") or "FastAPI Agent Workflow 调用失败。")

    data = api_result.get("data") or {}
    full_report = data.get("analysis", "")
    if not full_report.strip():
        return _analysis_error_result("FastAPI Agent Workflow 未返回分析文本。")

    chunks = data.get("retrieved_chunks", [])
    trace = data.get("trace") or {}
    return {
        **_split_analysis_report(full_report),
        "success": True,
        "error": data.get("error"),
        "retrieved_chunks": chunks,
        "rag_sources": chunks,
        "retrieved_chunk_count": len(chunks),
        "rag_top_k": top_k,
        "rag_total_chunks": len(chunks),
        "workflow_steps": data.get("workflow_steps", []),
        "trace": trace,
        "used_rerank": trace.get("used_rerank", False),
        "rerank_method": trace.get("rerank_method"),
        "llm_provider": data.get("llm_provider") or trace.get("llm_provider"),
        "llm_model": data.get("llm_model") or trace.get("llm_model"),
        "use_mock_llm": trace.get("use_mock_llm", False),
        "fallback_to_mock": trace.get("fallback_to_mock", True),
        "fallback_used": data.get("fallback_used", trace.get("fallback_used", False)),
        "original_provider": data.get("original_provider") or trace.get("original_provider"),
        "provider_error": data.get("provider_error") or trace.get("provider_error"),
        "api_mode": True,
        "api_used_rag": use_rag,
    }


def _run_api_rag_analysis(
    base_url: str,
    resume_text: str,
    job_description: str,
    top_k: int,
    use_rerank: bool,
    llm_provider: str,
    llm_model: str | None = None,
    fallback_to_mock: bool = True,
) -> dict:
    retrieval_call = call_rag_retrieve_api(
        base_url=base_url,
        resume_text=resume_text,
        job_description=job_description,
        top_k=top_k,
        use_rerank=use_rerank,
    )
    if not retrieval_call.get("success"):
        return _analysis_error_result(retrieval_call.get("error") or "FastAPI RAG 检索失败。")

    retrieval_data = retrieval_call.get("data") or {}
    chunks = retrieval_data.get("retrieved_chunks", [])
    if not chunks:
        return _analysis_error_result("FastAPI RAG 检索未返回可用于分析的片段。")

    llm_result = llm_match_analysis_tool(
        resume_text=resume_text,
        job_description=job_description,
        retrieved_chunks=chunks,
        use_rag=True,
        llm_provider=llm_provider,
        llm_model=llm_model,
        use_mock_llm=llm_provider == "mock",
        fallback_to_mock=fallback_to_mock,
    )
    if not llm_result.success:
        return _analysis_error_result(llm_result.error or "本地 LLM 分析失败。")

    return {
        **llm_result.data,
        "success": True,
        "error": None,
        "retrieved_chunks": chunks,
        "rag_sources": chunks,
        "retrieved_chunk_count": len(chunks),
        "rag_top_k": top_k,
        "rag_total_chunks": len(chunks),
        "used_rerank": retrieval_data.get("used_rerank", False),
        "rerank_method": retrieval_data.get("rerank_method"),
        "api_mode": True,
        "api_hybrid_mode": True,
    }


def load_sample_text(path: Path) -> str:
    """读取示例文件；如果文件缺失，返回友好提示文本。"""
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return f"示例文件不存在：{path}"


def load_demo_data() -> None:
    """将示例简历和岗位 JD 写入 session_state，供分析页直接使用。"""
    st.session_state["job_description_input"] = load_sample_text(SAMPLE_JOB_PATH)
    st.session_state["resume_text_input"] = load_sample_text(SAMPLE_RESUME_PATH)
    st.session_state["demo_mode_enabled"] = True


def build_export_report(result: dict, rag_enabled: bool) -> str:
    """将分析结果整理为 Markdown，供 Streamlit 下载。"""
    sections = [
        "# AI 简历与岗位匹配分析报告",
        "",
        result.get("job_analysis", ""),
        "",
        result.get("resume_analysis", ""),
        "",
        result.get("match_analysis", ""),
        "",
        result.get("suggestions", ""),
    ]

    if rag_enabled and result.get("rag_sources"):
        top_k = result.get("rag_top_k", len(result.get("rag_sources", [])))
        actual_count = len(result.get("rag_sources", []))
        total_chunks = result.get("rag_total_chunks", actual_count)
        section_filter = result.get("rag_section_filter")
        section_label = SECTION_LABELS.get(section_filter, section_filter or "全部")
        sections.extend(
            [
                "",
                "## RAG 检索说明",
                "",
                f"本次 RAG 检索召回 top_k = {top_k} 个片段。",
                f"- 本次简历生成 chunk 数量：{total_chunks}",
                f"- 用户设置 top_k：{top_k}",
                f"- 实际召回片段数量：{actual_count}",
                f"- 检索范围 section：{section_label}",
                "以下片段作为大模型分析的参考上下文。",
                "",
            ]
        )
        if actual_count < top_k:
            sections.extend(
                [
                    "实际召回数量少于 top_k，原因通常是可检索 chunk 总数不足。",
                    "",
                ]
            )
        for index, source in enumerate(result["rag_sources"], start=1):
            distance = source.get("distance")
            distance_text = f"{distance:.4f}" if distance is not None else "未返回"
            preview = source.get("text", "")[:300]
            if len(source.get("text", "")) > 300:
                preview += "……"

            sections.extend(
                [
                    f"### 片段 {index}",
                    f"- chunk_id：{source.get('chunk_id', source.get('chunk_index', index))}",
                    f"- source：{source.get('source', 'resume')}",
                    f"- file_name：{source.get('file_name', source.get('source_name', '未知来源'))}",
                    f"- chunk_length：{source.get('chunk_length', len(source.get('text', '')))}",
                    f"- section：{source.get('section', 'unknown')}",
                    f"- distance：{distance_text}",
                    "",
                    f"内容预览：{preview}",
                    "",
                ]
            )

    return "\n".join(sections).strip() + "\n"


def render_agent_trace(result: dict) -> None:
    """Render the lightweight Agent Workflow trace and JSON export."""
    trace = result.get("trace") or {}
    if not trace:
        return

    render_section_title("Trace 运行摘要")
    summary_columns = st.columns(4)
    summary_columns[0].metric("run_id", trace.get("run_id", "-"))
    duration = trace.get("duration_ms")
    summary_columns[1].metric("总耗时", f"{duration:.2f} ms" if duration is not None else "-")
    summary_columns[2].metric("final_status", trace.get("final_status", "-"))
    summary_columns[3].metric("fallback_used", format_bool(trace.get("fallback_used")))

    detail_columns = st.columns(4)
    detail_columns[0].metric("llm_provider", trace.get("llm_provider") or "-")
    detail_columns[1].metric("llm_model", trace.get("llm_model") or "-")
    detail_columns[2].metric("review_passed", format_bool(trace.get("review_passed")))
    detail_columns[3].metric("retrieval_attempts", trace.get("retrieval_attempts", "-"))

    with st.expander("查看 Trace 运行上下文", expanded=False):
        st.write(
            {
                "mode": trace.get("mode"),
                "resume_length": trace.get("resume_length"),
                "job_description_length": trace.get("job_description_length"),
                "top_k": trace.get("top_k"),
                "used_rag": trace.get("used_rag"),
                "used_rerank": trace.get("used_rerank"),
                "rerank_method": trace.get("rerank_method"),
                "embedding_provider": trace.get("embedding_provider"),
                "used_fallback": trace.get("used_fallback"),
                "use_mock_llm": trace.get("use_mock_llm"),
                "fallback_to_mock": trace.get("fallback_to_mock"),
                "original_provider": trace.get("original_provider"),
                "provider_error": trace.get("provider_error"),
            }
        )

    render_section_title("Trace Steps 明细")
    for index, step in enumerate(trace.get("steps", []), start=1):
        status = "成功" if step.get("success") else "失败"
        title = f"Step {index}: {step.get('step_name', '')} / {step.get('tool_name', '')} / {status}"
        with st.expander(title):
            step_columns = st.columns(3)
            step_columns[0].metric("success", status)
            step_columns[1].metric("duration_ms", f"{step.get('duration_ms', 0):.2f}")
            step_columns[2].metric("tool", step.get("tool_name", "-"))
            st.write(f"message：{step.get('message', '')}")
            st.markdown("**input_summary**")
            st.json(step.get("input_summary") or {})
            st.markdown("**output_summary**")
            st.json(step.get("output_summary") or {})
            if step.get("error"):
                st.error(step["error"])

    with st.expander("Trace JSON", expanded=False):
        st.json(trace)
    trace_json = json.dumps(trace, ensure_ascii=False, indent=2)
    run_id = trace.get("run_id", "unknown")
    st.download_button(
        "下载 Trace JSON",
        data=trace_json,
        file_name=f"trace_{run_id}.json",
        mime="application/json",
        key=f"download_trace_{run_id}",
    )

    if result.get("trace_save_error"):
        st.warning(f"Trace 已在页面生成，但保存到本地失败：{result['trace_save_error']}")
    elif result.get("trace_path"):
        st.caption(f"Trace 已保存：{result['trace_path']}")


def render_result_summary_cards(
    result: dict,
    analysis_mode: str,
    selected_llm_provider: str,
    result_rag_enabled: bool,
    result_agent_workflow_enabled: bool,
) -> None:
    review_result = result.get("review_result") or {}
    review_passed = review_result.get("review_passed")
    columns = st.columns(3)
    columns[0].metric("当前模式", analysis_mode)
    columns[1].metric("LLM Provider", result.get("llm_provider") or selected_llm_provider)
    columns[2].metric("使用 RAG", format_bool(result_rag_enabled or result.get("api_used_rag")))
    columns = st.columns(3)
    columns[0].metric("使用 Rerank", format_bool(result.get("used_rerank")))
    columns[1].metric("Query Refinement", format_bool(result.get("query_refinement_used")))
    columns[2].metric("Reviewer", format_bool(review_passed) if result_agent_workflow_enabled else "-")


def render_reviewer_result(review_result: dict) -> None:
    if not review_result:
        return

    review_passed = review_result.get("review_passed")
    if review_passed is True:
        render_status_card("Reviewer 审核通过", review_result.get("review_summary", ""), "success")
    elif review_passed is False:
        render_status_card("Reviewer 审核未通过", review_result.get("review_summary", ""), "warning")
    else:
        render_status_card("Reviewer 未执行", "本次结果没有返回 reviewer 审核状态。", "info")

    with st.expander("查看 Reviewer 审核明细", expanded=(review_passed is False)):
        missing = review_result.get("missing_points", [])
        risks = review_result.get("risk_notes", [])
        evidence = review_result.get("evidence_usage", "")
        suggestions = review_result.get("improvement_suggestions", [])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**missing_points**")
            if missing:
                for point in missing:
                    st.markdown(f"- {point}")
            else:
                st.caption("未发现明显缺失项。")
            st.markdown("**evidence_usage**")
            st.write(evidence or "未返回")
        with col2:
            st.markdown("**risk_notes**")
            if risks:
                for risk in risks:
                    st.markdown(f"- {risk}")
            else:
                st.caption("未返回风险提示。")
            st.markdown("**improvement_suggestions**")
            if suggestions:
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
            else:
                st.caption("未返回改进建议。")


def render_rag_sources(rag_sources: list[dict]) -> None:
    with st.expander("查看 RAG 检索片段", expanded=True):
        show_full_rag_chunks = st.checkbox(
            "显示完整 RAG 片段内容",
            value=False,
            key="show_full_rag_chunks",
        )

        for index, source in enumerate(rag_sources, start=1):
            chunk_text = source.get("text") or source.get("content") or ""
            preview_text = chunk_text
            if not show_full_rag_chunks and len(chunk_text) > 420:
                preview_text = f"{chunk_text[:420]}……"
            distance = source.get("distance")
            distance_text = f"{distance:.4f}" if distance is not None else "未返回"
            rerank_score = source.get("rerank_score")
            rerank_text = f"{rerank_score:.4f}" if rerank_score is not None else "-"
            keyword_hits = source.get("keyword_hits", [])
            keyword_text = ", ".join(str(item) for item in keyword_hits) if keyword_hits else "-"
            section_bonus = source.get("section_bonus")
            section_bonus_text = f"{section_bonus:.4f}" if section_bonus is not None else "-"
            section = escape(str(source.get("section", "unknown")))
            chunk_id = escape(str(source.get("chunk_id", source.get("chunk_index", index))))
            chunk_length = escape(str(source.get("chunk_length", len(chunk_text))))

            st.markdown(
                f"""
                <div class="chunk-card">
                    <div>
                        <span class="badge-pill">片段 {index}</span>
                        <span class="badge-pill">section: {section}</span>
                    </div>
                    <div class="chunk-meta">
                        chunk_id: {chunk_id}
                        · chunk_length: {chunk_length}
                        · distance: {escape(distance_text)}
                        · rerank_score: {escape(rerank_text)}
                        · section_bonus: {escape(section_bonus_text)}
                    </div>
                    <div class="chunk-meta">keyword_hits: {escape(keyword_text)}</div>
                    <div class="chunk-preview">{escape(preview_text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_workflow_steps(steps: list[dict]) -> None:
    with st.expander("查看 Agent Workflow Steps", expanded=True):
        for index, step in enumerate(steps, start=1):
            status = "成功" if step.get("success") else "失败"
            render_status_card(
                f"Step {index}: {step.get('step_name', '')}",
                f"tool_name：{step.get('tool_name', '')} · success：{status} · message：{step.get('message', '')}",
                "success" if step.get("success") else "warning",
            )
            data_summary = step.get("data_summary") or {}
            if data_summary:
                st.json(data_summary)
            if step.get("error"):
                st.error(step["error"])


def read_txt_file(file_bytes: bytes) -> str:
    """
    读取 txt 文件内容。
    优先使用 UTF-8，如果失败再尝试 GBK，兼容中文 Windows 文本。
    """
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return file_bytes.decode("gbk")
        except UnicodeDecodeError:
            return "文件解码失败，请确认 txt 文件编码是否正常。"


def read_pdf_file(file_bytes: bytes) -> str:
    """
    读取 PDF 文件中的文本内容。
    注意：仅支持可复制文字的 PDF，不支持扫描版图片 PDF。
    """
    try:
        reader = PdfReader(BytesIO(file_bytes))
        text_list = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_list.append(page_text)

        text = "\n".join(text_list).strip()

        if not text:
            return "PDF 未读取到有效文本，可能是扫描版 PDF 或图片型 PDF。"

        return text

    except Exception as e:
        return f"PDF 文件读取失败：{e}"


def read_docx_file(file_bytes: bytes) -> str:
    """
    读取 docx 文件中的段落文本。
    注意：当前版本支持 .docx，不支持老版 .doc。
    """
    try:
        document = Document(BytesIO(file_bytes))
        text_list = []

        for paragraph in document.paragraphs:
            paragraph_text = paragraph.text.strip()
            if paragraph_text:
                text_list.append(paragraph_text)

        text = "\n".join(text_list).strip()

        if not text:
            return "Word 文件未读取到有效文本。"

        return text

    except Exception as e:
        return f"Word 文件读取失败：{e}"


def read_uploaded_resume_file(uploaded_file) -> str:
    """
    根据文件类型读取上传的简历内容。
    支持 txt、pdf、docx。
    """
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if file_name.endswith(".txt"):
        return read_txt_file(file_bytes)

    if file_name.endswith(".pdf"):
        return read_pdf_file(file_bytes)

    if file_name.endswith(".docx"):
        return read_docx_file(file_bytes)

    return "暂不支持该文件格式，请上传 txt、pdf 或 docx 文件。"


def render_home_page() -> None:
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title">AI Resume Agent</div>
            <div class="hero-subtitle">简历与岗位匹配分析助手</div>
            <div class="hero-desc">
                一个面向 AI 应用开发求职场景的简历与岗位匹配分析工具，
                支持 RAG 检索增强、Agent Workflow、Lightweight Harness、Trace 可观测和 Eval 评测。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_badges(
        [
            APP_VERSION_LABEL,
            "RAG + Rerank",
            "Agent Workflow",
            "Reviewer Agent",
            "Gemini / DeepSeek Ready",
            "FastAPI Mode",
            "Trace + Eval",
        ]
    )
    render_status_card(
        "项目定位",
        "当前项目定位是 AI 应用工程化 MVP / RAG Workflow 原型 / Lightweight Agent Harness Demo，"
        "适合学习和面试展示，不是完整生产级 Agent 平台。",
        "info",
    )

    render_section_title("核心能力")
    feature_cards = [
        ("简历解析", "支持 txt / pdf / docx 简历文本读取，为后续分析和检索提供统一输入。"),
        ("RAG 检索增强", "基于 ChromaDB + local_bge 等 embedding provider 召回与岗位 JD 相关的简历片段。"),
        ("Lightweight Rerank", "在向量召回后使用关键词、section bonus 和 distance 做轻量二次排序。"),
        ("Agent Workflow", "将 RAG 检索、LLM 分析、报告生成等能力封装为固定工具调用链路。"),
        ("Reviewer Agent", "对生成报告做规则化审核，提示缺失结构、证据使用和潜在风险。"),
        ("Trace / Observability", "记录 workflow steps、Provider 状态、耗时、fallback 和错误摘要。"),
        ("Eval Runner", "使用 Recall@K、MRR、section / keyword hit 等指标验证工程链路稳定性。"),
        ("Multi-Model Provider", "支持 Gemini、DeepSeek、OpenAI Compatible 和 Mock Provider，便于演示与降级。"),
    ]
    for row_start in range(0, len(feature_cards), 4):
        columns = st.columns(4)
        for column, (title, text) in zip(columns, feature_cards[row_start : row_start + 4]):
            with column:
                render_feature_card(title, text)

    render_section_title("推荐演示路径")
    step_columns = st.columns(3)
    demo_steps = [
        ("Step 1", "进入示例演示，加载脱敏样例。"),
        ("Step 2", "进入简历岗位匹配分析，选择 Agent Workflow，并启用 RAG + Rerank。"),
        ("Step 3", "查看 RAG 召回片段、Reviewer 审核结果、Trace JSON 和 Markdown 报告。"),
    ]
    for column, (title, text) in zip(step_columns, demo_steps):
        with column:
            render_feature_card(title, text)


def render_analysis_page() -> None:
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title">简历岗位匹配分析</div>
            <div class="hero-subtitle">从 JD 到证据召回，再到 Agent Workflow 报告</div>
            <div class="hero-desc">
                支持普通 LLM 基线、RAG 检索增强和 Agent Workflow 三种模式。
                面试演示建议选择 Agent Workflow，并开启 RAG + 轻量 Rerank。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    backend_mode = st.session_state.get("backend_runtime_mode", LOCAL_BACKEND_MODE)
    api_base_url = st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)
    selected_llm_provider = st.session_state.get(
        "selected_llm_provider",
        get_llm_provider_from_env(),
    )
    fallback_to_mock = st.session_state.get("fallback_to_mock", True)
    current_embedding_provider = get_embedding_provider()
    render_badges(
        [
            f"运行模式：{backend_mode}",
            f"LLM：{LLM_PROVIDER_LABELS.get(selected_llm_provider, selected_llm_provider)}",
            f"Embedding：{current_embedding_provider}",
            "RAG 可解释召回",
            "Trace 可下载",
        ]
    )
    render_status_card(
        "RAG 说明",
        "RAG 模式会先从简历中检索与岗位最相关的片段，再交给大模型生成分析结果；"
        "如果在线 embedding 不可用，项目会按已有 fallback 逻辑使用本地向量。",
        "info",
    )
    if current_embedding_provider == "local_bge":
        st.caption("首次使用本地语义模型可能需要下载模型，耗时较长；下载完成后会使用本地缓存。")

    if st.session_state.get("demo_mode_enabled"):
        st.info("当前已加载示例数据。你仍然可以上传自己的简历文件，或直接编辑下方文本框内容。")

    render_section_title("输入材料")
    input_col1, input_col2 = st.columns(2)
    with input_col1:
        st.markdown('<div class="soft-card"><strong>岗位 JD 输入</strong><br>粘贴目标岗位描述，用于匹配分析和 RAG query。</div>', unsafe_allow_html=True)
        job_description = st.text_area(
            "请输入岗位描述",
            height=260,
            placeholder="例如：粘贴 AI 应用开发实习生的岗位描述。",
            key="job_description_input",
        )
    with input_col2:
        st.markdown('<div class="soft-card"><strong>简历输入 / 上传</strong><br>上传文件优先；也可以直接粘贴项目经历文本。</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "上传个人简历 / 经历文件",
            type=["txt", "pdf", "docx"],
            help="当前支持 txt、pdf、docx 文件。PDF 需为可复制文字的文本型 PDF。",
        )
        resume_text_input = st.text_area(
            "请输入个人简历或项目经历",
            height=220,
            placeholder="如果没有上传文件，可以在这里粘贴你的专业技能、项目经历和自我评价内容。",
            key="resume_text_input",
        )

    uploaded_resume_text = ""
    uploaded_source_name = ""

    if uploaded_file is not None:
        uploaded_source_name = uploaded_file.name
        uploaded_resume_text = read_uploaded_resume_file(uploaded_file)

        if (
            uploaded_resume_text.startswith("文件解码失败")
            or uploaded_resume_text.startswith("PDF 文件读取失败")
            or uploaded_resume_text.startswith("Word 文件读取失败")
            or uploaded_resume_text.startswith("PDF 未读取到")
            or uploaded_resume_text.startswith("Word 文件未读取到")
            or uploaded_resume_text.startswith("暂不支持")
        ):
            st.error(uploaded_resume_text)
            uploaded_resume_text = ""
        else:
            st.success("文件读取成功，系统将优先使用上传文件内容作为个人经历。")

            with st.expander("查看上传文件内容预览"):
                st.write(uploaded_resume_text[:1500])

    final_resume_text = uploaded_resume_text.strip() if uploaded_resume_text.strip() else resume_text_input.strip()
    source_name = uploaded_source_name if uploaded_resume_text.strip() else "手动输入内容"

    render_section_title("选择分析模式")
    mode_cols = st.columns(3)
    with mode_cols[0]:
        render_feature_card("普通 LLM 分析", "适合作为基线对比，直接将简历和 JD 交给模型分析。")
    with mode_cols[1]:
        render_feature_card("RAG 检索增强分析", "适合查看证据召回，能展示模型参考了哪些简历片段。")
    with mode_cols[2]:
        render_feature_card("Agent Workflow 分析", "推荐演示模式，包含工具调用链、Reviewer 和 Trace。")
    analysis_mode = st.radio(
        "分析模式",
        ["普通 LLM 分析", "RAG 检索增强分析", "Agent Workflow 分析"],
        horizontal=True,
        help="Agent Workflow 分析会按固定工具调用链路执行：RAG 检索工具 -> LLM 分析工具。",
    )
    rag_enabled = analysis_mode in ["RAG 检索增强分析", "Agent Workflow 分析"]
    agent_workflow_enabled = analysis_mode == "Agent Workflow 分析"
    rag_top_k = 3
    if rag_enabled:
        rag_top_k = st.slider("RAG 召回片段数量 top_k", min_value=1, max_value=8, value=3)
        st.caption("top_k 表示从向量数据库中召回最相关的前 k 个简历片段。")
        use_rerank = st.checkbox(
            "启用轻量 Rerank",
            value=False,
            help="在 ChromaDB 初步召回后，根据 JD 关键词、section 和 distance 二次排序。",
        )
        if backend_mode == API_BACKEND_MODE:
            section_filter = None
            st.caption("FastAPI MVP 暂未暴露 section_filter，API 模式使用全部简历片段检索。")
        else:
            section_label = st.selectbox("RAG 检索范围", list(SECTION_OPTIONS.keys()), index=0)
            section_filter = SECTION_OPTIONS[section_label]
    else:
        section_filter = None
        use_rerank = False

    selected_llm_model = st.session_state.get("selected_llm_model")

    if st.button("开始分析"):
        if not job_description.strip() or not final_resume_text:
            st.warning("请先输入岗位描述，并上传或填写个人经历。")
        else:
            with st.spinner("正在分析中，请稍等..."):
                if backend_mode == API_BACKEND_MODE:
                    if agent_workflow_enabled:
                        result = _agent_api_result(
                            call_agent_workflow_api(
                                base_url=api_base_url,
                                resume_text=final_resume_text,
                                job_description=job_description,
                                top_k=rag_top_k,
                                use_rag=True,
                                use_rerank=use_rerank,
                                llm_provider=selected_llm_provider,
                                llm_model=selected_llm_model,
                                use_mock_llm=selected_llm_provider == "mock",
                                fallback_to_mock=fallback_to_mock,
                            ),
                            top_k=rag_top_k,
                            use_rag=True,
                        )
                    elif rag_enabled:
                        result = _run_api_rag_analysis(
                            base_url=api_base_url,
                            resume_text=final_resume_text,
                            job_description=job_description,
                            top_k=rag_top_k,
                            use_rerank=use_rerank,
                            llm_provider=selected_llm_provider,
                            llm_model=selected_llm_model,
                            fallback_to_mock=fallback_to_mock,
                        )
                    else:
                        result = _agent_api_result(
                            call_agent_workflow_api(
                                base_url=api_base_url,
                                resume_text=final_resume_text,
                                job_description=job_description,
                                top_k=rag_top_k,
                                use_rag=False,
                                use_rerank=False,
                                llm_provider=selected_llm_provider,
                                llm_model=selected_llm_model,
                                use_mock_llm=selected_llm_provider == "mock",
                                fallback_to_mock=fallback_to_mock,
                            ),
                            top_k=rag_top_k,
                            use_rag=False,
                        )
                else:
                    if agent_workflow_enabled:
                        result = run_resume_agent_workflow(
                            resume_text=final_resume_text,
                            job_description=job_description,
                            top_k=rag_top_k,
                            use_rag=True,
                            source_name=source_name,
                            section_filter=section_filter,
                            use_rerank=use_rerank,
                            llm_provider=selected_llm_provider,
                            llm_model=selected_llm_model,
                            use_mock_llm=selected_llm_provider == "mock",
                            fallback_to_mock=fallback_to_mock,
                        )
                    elif rag_enabled:
                        result = run_rag_workflow(
                            job_description,
                            final_resume_text,
                            source_name=source_name,
                            top_k=rag_top_k,
                            section_filter=section_filter,
                            use_rerank=use_rerank,
                            llm_provider=selected_llm_provider,
                            llm_model=selected_llm_model,
                            use_mock_llm=selected_llm_provider == "mock",
                            fallback_to_mock=fallback_to_mock,
                        )
                    else:
                        result = run_agent_workflow(
                            job_description,
                            final_resume_text,
                            llm_provider=selected_llm_provider,
                            llm_model=selected_llm_model,
                            use_mock_llm=selected_llm_provider == "mock",
                            fallback_to_mock=fallback_to_mock,
                        )

            # 保存本次分析结果，避免点击下载按钮或切换 RAG 片段预览时页面 rerun 后结果消失。
            st.session_state["last_analysis_result"] = result
            st.session_state["last_analysis_rag_enabled"] = rag_enabled
            st.session_state["last_analysis_agent_workflow_enabled"] = agent_workflow_enabled
            st.session_state["last_analysis_mode"] = analysis_mode

    result = st.session_state.get("last_analysis_result")
    result_rag_enabled = st.session_state.get("last_analysis_rag_enabled", False)
    result_agent_workflow_enabled = st.session_state.get("last_analysis_agent_workflow_enabled", False)
    result_analysis_mode = st.session_state.get("last_analysis_mode", analysis_mode)

    if result:
        if result.get("error"):
            st.error(result["error"])
        else:
            render_section_title("分析结果总览")
            st.success("分析完成")
            render_result_summary_cards(
                result,
                result_analysis_mode,
                selected_llm_provider,
                result_rag_enabled,
                result_agent_workflow_enabled,
            )
            st.caption(
                f"LLM：{result.get('llm_provider', selected_llm_provider)}"
                f" / {result.get('llm_model') or 'default model'}"
            )
            if result.get("fallback_used"):
                st.warning(
                    f"真实 Provider {result.get('original_provider')} 调用失败，已 fallback 到 Mock。"
                    f"错误摘要：{result.get('provider_error') or '未返回'}"
                )
            if result.get("api_hybrid_mode"):
                st.info("RAG retrieve API 已完成；当前 API 尚无独立 RAG 分析接口，LLM 分析仍走本地逻辑。")
            if result_agent_workflow_enabled and result.get("workflow_steps"):
                render_workflow_steps(result["workflow_steps"])

            # --- Harness: review result ---
            review_result = result.get("review_result") or {}
            render_reviewer_result(review_result)

            # --- Harness: query refinement ---
            if result.get("query_refinement_used"):
                st.info(
                    f"初次 RAG 检索质量不足，已自动重试（共 {result.get('retrieval_attempts', 2)} 次检索）。"
                )

            if result_rag_enabled and result.get("retrieved_chunk_count") is not None:
                actual_count = result.get("retrieved_chunk_count", 0)
                top_k = result.get("rag_top_k", actual_count)
                total_chunks = result.get("rag_total_chunks", actual_count)

                st.info(
                    f"本次简历共生成 {total_chunks} 个 chunk。"
                    f"用户设置 top_k = {top_k}。"
                    f"实际召回 {actual_count} 个片段。"
                )

                if actual_count < top_k:
                    if result.get("rag_available_filtered_chunks") == 0:
                        st.warning("当前检索范围没有可召回的 chunk，请切换为“全部”或选择其他简历模块。")
                    else:
                        st.warning("实际召回数量少于 top_k，通常是因为当前简历切分后的 chunk 总数不足。")

            rag_sources = result.get("rag_sources", [])
            if result_rag_enabled and rag_sources:
                render_rag_sources(rag_sources)

            report_markdown = build_export_report(result, rag_enabled=result_rag_enabled)
            st.download_button(
                "下载分析报告 Markdown",
                data=report_markdown,
                file_name="resume_match_report.md",
                mime="text/markdown",
            )

        if result_agent_workflow_enabled:
            render_agent_trace(result)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["岗位要求分析", "个人能力分析", "匹配度分析", "简历优化建议"]
        )

        with tab1:
            st.markdown(result["job_analysis"])

        with tab2:
            st.markdown(result["resume_analysis"])

        with tab3:
            st.markdown(result["match_analysis"])

        with tab4:
            st.markdown(result["suggestions"])


def render_demo_page() -> None:
    st.title("示例演示")
    st.write("这里提供一组脱敏示例数据，方便快速演示完整项目流程。")

    if st.button("加载示例数据并进入分析"):
        load_demo_data()
        st.session_state["target_page"] = "简历岗位匹配分析"
        st.rerun()

    st.subheader("示例岗位 JD")
    st.text_area(
        "sample_job_description.txt",
        value=load_sample_text(SAMPLE_JOB_PATH),
        height=260,
        disabled=True,
    )

    st.subheader("示例简历")
    st.text_area(
        "sample_resume.txt",
        value=load_sample_text(SAMPLE_RESUME_PATH),
        height=320,
        disabled=True,
    )


def render_rag_detail_page() -> None:
    st.title("RAG 召回片段说明")
    st.write("本页用于解释项目中的 RAG 检索增强流程，帮助面试讲解和调试。")

    st.subheader("RAG 流程")
    steps = [
        "解析简历文本",
        "文本清洗和切分 chunk",
        "生成 embedding",
        "存入 ChromaDB",
        "将岗位 JD 转成 query",
        "检索 top_k 相关简历片段",
        "将召回片段拼入 Prompt",
        "调用大模型生成分析结果",
    ]
    for index, step in enumerate(steps, start=1):
        st.markdown(f"{index}. {step}")

    st.subheader("关键概念")
    st.markdown("- chunk：简历文本切分后的片段，是 RAG 检索的基本单位。")
    st.markdown("- overlap：相邻 chunk 的重叠部分，用来减少上下文被截断的问题。")
    st.markdown("- top_k：从向量数据库中召回最相关的前 k 个片段。")
    st.markdown("- metadata：记录来源文件名、chunk_index、embedding_provider 等信息，便于解释和调试。")

    st.info("分析完成后，可以在结果区展开“查看 RAG 检索片段”，查看模型参考了哪些简历内容。")


def render_about_page() -> None:
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title">关于项目 / 后续计划</div>
            <div class="hero-subtitle">一个可运行、可验证、可讲解的 AI 应用工程化 Demo</div>
            <div class="hero-desc">
                当前版本重点展示 RAG Workflow、Tool Calling、Lightweight Agent Harness、
                Trace、Eval 和 FastAPI 接口雏形，不包装成生产级平台。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_title("项目已完成能力")
    completed = [
        ("输入解析", "txt / pdf / docx 简历解析，支持示例数据快速演示。"),
        ("RAG Workflow", "section-aware chunking、ChromaDB 检索、local / local_bge / gemini embedding provider。"),
        ("Agent Harness", "Tool Calling、Agent Workflow、Reviewer Agent、Bounded Query Refinement Loop。"),
        ("可观测与评测", "Trace JSON、Workflow Steps、Eval Runner、RAG Evaluation Metrics。"),
        ("模型与接口", "Gemini / DeepSeek / OpenAI Compatible / Mock Provider，支持 fallback_to_mock。"),
        ("Demo 工程化", "Streamlit Demo、FastAPI Backend、local mode / API mode、Markdown 报告导出。"),
    ]
    for row_start in range(0, len(completed), 3):
        columns = st.columns(3)
        for column, (title, text) in zip(columns, completed[row_start : row_start + 3]):
            with column:
                render_feature_card(title, text)

    render_section_title("当前边界")
    boundary_cols = st.columns(2)
    boundaries = [
        "没有用户登录",
        "没有数据库历史记录",
        "没有云部署",
        "没有生产级权限控制",
        "没有复杂 autonomous planning",
        "没有沙盒执行环境",
    ]
    for index, item in enumerate(boundaries):
        with boundary_cols[index % 2]:
            render_status_card(item, "当前实现保持 MVP 范围，用于本地演示和面试讲解。", "warning")

    render_section_title("后续方向")
    future_items = [
        ("Hybrid Search", "结合关键词检索和向量检索，提升召回稳定性。"),
        ("Cross-encoder Rerank", "用更强的 reranker 替代当前 rule-based rerank。"),
        ("More Eval Cases", "增加更多岗位、简历样例和 gold evidence。"),
        ("Docker Deployment", "补充可复现部署环境。"),
        ("Memory MVP", "保存用户偏好、历史 JD 和报告版本。"),
        ("Sandbox / Permission Control", "为更复杂工具调用增加权限和执行边界。"),
    ]
    for row_start in range(0, len(future_items), 3):
        columns = st.columns(3)
        for column, (title, text) in zip(columns, future_items[row_start : row_start + 3]):
            with column:
                render_feature_card(title, text)


st.set_page_config(
    page_title="AI Resume Agent",
    page_icon="🤖",
    layout="wide",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.header("AI Resume Agent")
    st.markdown(
        f"""
        <div class="sidebar-footer">
            <strong>当前版本</strong><br>{APP_VERSION_LABEL}<br><br>
            建议面试演示使用 Agent Workflow + RAG + Rerank。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("功能入口")
    page_options = [
        "首页 / 项目介绍",
        "简历岗位匹配分析",
        "示例演示",
        "RAG 召回片段说明",
        "关于项目 / 后续计划",
    ]
    target_page = st.session_state.pop("target_page", None)
    if target_page in page_options:
        st.session_state["selected_page"] = target_page
    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = page_options[0]

    page = st.radio(
        "请选择页面",
        page_options,
        key="selected_page",
    )

    st.divider()
    st.subheader("运行后端模式")
    backend_mode = st.radio(
        "选择调用方式",
        [LOCAL_BACKEND_MODE, API_BACKEND_MODE],
        key="backend_runtime_mode",
        help="本地模式直接调用 Python；API 模式通过 HTTP 调用 FastAPI。",
    )
    st.caption(f"当前运行模式：{backend_mode}")
    if backend_mode == API_BACKEND_MODE:
        if "api_base_url" not in st.session_state:
            st.session_state["api_base_url"] = DEFAULT_API_BASE_URL
        api_base_url = st.text_input(
            "API Base URL",
            key="api_base_url",
        )
        if st.button("检查 API 连接", key="check_api_connection"):
            health_result = check_api_health(api_base_url)
            if health_result.get("success"):
                health_data = health_result.get("data") or {}
                st.success(
                    f"API 连接成功：{health_data.get('project', 'FastAPI')} "
                    f"{health_data.get('version', '')}"
                )
            else:
                st.error(health_result.get("error") or "FastAPI 连接失败。")

    st.subheader("LLM Provider")
    if "selected_llm_provider" not in st.session_state:
        st.session_state["selected_llm_provider"] = get_llm_provider_from_env()

    provider_labels = [LLM_PROVIDER_LABELS[p] for p in LLM_PROVIDER_OPTIONS]
    label_to_provider = {v: k for k, v in LLM_PROVIDER_LABELS.items()}
    # Sync label index from internal value (survives page reruns)
    current_internal = st.session_state["selected_llm_provider"]
    current_label = LLM_PROVIDER_LABELS.get(current_internal, provider_labels[0])
    if "provider_label_idx" not in st.session_state:
        try:
            st.session_state["provider_label_idx"] = provider_labels.index(current_label)
        except ValueError:
            st.session_state["provider_label_idx"] = 0

    selected_label = st.selectbox(
        "选择模型调用方式",
        provider_labels,
        key="provider_label_idx",
    )
    selected_llm_provider = label_to_provider[selected_label]
    st.session_state["selected_llm_provider"] = selected_llm_provider
    st.caption(f"当前模型：{selected_label}")

    # DeepSeek model selector
    selected_llm_model: str | None = None
    if selected_llm_provider == "deepseek":
        default_deepseek = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        if "deepseek_model_choice" not in st.session_state:
            st.session_state["deepseek_model_choice"] = (
                default_deepseek if default_deepseek in DEEPSEEK_MODEL_OPTIONS else "deepseek-v4-flash"
            )
        model_choice = st.selectbox(
            "DeepSeek 模型",
            DEEPSEEK_MODEL_OPTIONS,
            key="deepseek_model_choice",
        )
        if model_choice == "custom":
            custom_model = st.text_input(
                "自定义 DeepSeek 模型名",
                value=st.session_state.get("deepseek_custom_model", ""),
                key="deepseek_custom_model",
                placeholder="例如：deepseek-reasoner",
            )
            selected_llm_model = custom_model.strip() if custom_model.strip() else None
        else:
            selected_llm_model = model_choice
        st.caption(f"当前 DeepSeek 模型：{selected_llm_model or default_deepseek}")
    elif selected_llm_provider == "openai_compatible":
        st.info(
            "需要在 .env 中配置 OPENAI_COMPATIBLE_API_KEY、"
            "OPENAI_COMPATIBLE_BASE_URL 和 OPENAI_COMPATIBLE_MODEL。"
        )
    elif selected_llm_provider == "mock":
        st.warning("当前使用 mock LLM，只用于测试流程，不代表真实模型质量。")

    st.session_state["selected_llm_model"] = selected_llm_model

    if st.button("检查 LLM Provider", key="check_llm_provider"):
        if backend_mode == API_BACKEND_MODE:
            health_call = check_llm_health_api(
                api_base_url,
                provider=selected_llm_provider,
                model=selected_llm_model,
                use_mock=selected_llm_provider == "mock",
            )
            if health_call.get("success"):
                health_data = health_call.get("data") or {}
            else:
                health_data = {
                    "provider": selected_llm_provider,
                    "available": False,
                    "message": "Provider 健康检查请求失败。",
                    "error": health_call.get("error"),
                }
        else:
            health = check_llm_provider_health(
                provider=selected_llm_provider,
                model=selected_llm_model,
                use_mock=selected_llm_provider == "mock",
            )
            health_data = {
                "provider": health.provider,
                "model": health.model,
                "available": health.available,
                "latency_ms": health.latency_ms,
                "message": health.message,
                "error": health.error,
            }
        if health_data.get("available"):
            st.success(health_data.get("message") or "LLM Provider 可用。")
        else:
            st.error(health_data.get("error") or health_data.get("message") or "LLM Provider 不可用。")
        st.json(health_data)

    if "fallback_to_mock" not in st.session_state:
        st.session_state["fallback_to_mock"] = True
    fallback_to_mock = st.checkbox(
        "模型失败时 fallback 到 Mock",
        key="fallback_to_mock",
        help="用于演示和测试稳定性。真实生产环境应根据业务决定是否允许 fallback。",
    )

    st.divider()
    st.markdown(
        f"""
        <div class="sidebar-footer">
            <strong>Embedding Provider</strong><br>{get_embedding_provider()}<br><br>
            本项目是 AI 应用工程化 MVP，不是生产级 Agent 平台。
        </div>
        """,
        unsafe_allow_html=True,
    )

if page == "首页 / 项目介绍":
    render_home_page()
elif page == "简历岗位匹配分析":
    render_analysis_page()
elif page == "示例演示":
    render_demo_page()
elif page == "RAG 召回片段说明":
    render_rag_detail_page()
else:
    render_about_page()
