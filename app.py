import json
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
LLM_PROVIDER_OPTIONS = ["gemini", "openai_compatible", "mock"]


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

    st.subheader("Trace 运行摘要")
    summary_columns = st.columns(4)
    summary_columns[0].metric("run_id", trace.get("run_id", "-"))
    summary_columns[1].metric("mode", trace.get("mode", "-"))
    duration = trace.get("duration_ms")
    summary_columns[2].metric("总耗时", f"{duration:.2f} ms" if duration is not None else "-")
    summary_columns[3].metric("final_status", trace.get("final_status", "-"))

    st.write(
        {
            "resume_length": trace.get("resume_length"),
            "job_description_length": trace.get("job_description_length"),
            "top_k": trace.get("top_k"),
            "used_rag": trace.get("used_rag"),
            "used_rerank": trace.get("used_rerank"),
            "rerank_method": trace.get("rerank_method"),
            "embedding_provider": trace.get("embedding_provider"),
            "used_fallback": trace.get("used_fallback"),
            "llm_provider": trace.get("llm_provider"),
            "llm_model": trace.get("llm_model"),
            "use_mock_llm": trace.get("use_mock_llm"),
            "fallback_to_mock": trace.get("fallback_to_mock"),
            "fallback_used": trace.get("fallback_used"),
            "original_provider": trace.get("original_provider"),
            "provider_error": trace.get("provider_error"),
        }
    )

    st.subheader("Trace Steps 明细")
    for index, step in enumerate(trace.get("steps", []), start=1):
        title = f"Step {index}: {step.get('step_name', '')} / {step.get('tool_name', '')}"
        with st.expander(title):
            st.write(f"success：{'成功' if step.get('success') else '失败'}")
            st.write(f"duration_ms：{step.get('duration_ms', 0):.2f}")
            st.write(f"message：{step.get('message', '')}")
            st.markdown("**input_summary**")
            st.json(step.get("input_summary") or {})
            st.markdown("**output_summary**")
            st.json(step.get("output_summary") or {})
            if step.get("error"):
                st.error(step["error"])

    st.subheader("Trace JSON")
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
    st.title("AI Resume Agent / 简历与岗位匹配分析助手")
    st.write(
        "这是一个基于 Streamlit、Gemini API、ChromaDB 和 RAG Workflow 的简历与岗位匹配分析工具。"
        "用户可以上传简历并输入岗位 JD，系统会分析岗位要求、个人能力、匹配度和简历优化建议。"
    )
    st.info("当前项目定位是 AI 应用工程化 MVP / RAG Workflow 原型，不是完整生产级多工具 Agent。")

    st.subheader("当前能力")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("- txt / pdf / docx 简历解析")
        st.markdown("- 岗位 JD 输入")
        st.markdown("- 普通 LLM 分析")
    with col2:
        st.markdown("- RAG 检索增强分析")
        st.markdown("- ChromaDB 向量检索")
        st.markdown("- local / local_bge / gemini embedding")
    with col3:
        st.markdown("- RAG 召回片段查看")
        st.markdown("- Markdown 报告导出")
        st.markdown("- 示例数据演示")

    st.subheader("推荐演示路径")
    st.markdown("1. 进入“示例演示”，加载脱敏样例。")
    st.markdown("2. 进入“简历岗位匹配分析”，选择普通模式或 RAG 模式。")
    st.markdown("3. 查看四个分析 Tab、RAG 召回片段和 Markdown 导出报告。")


def render_analysis_page() -> None:
    st.title("简历岗位匹配分析")
    st.write("按步骤输入岗位 JD 和简历内容，可选择普通分析或 RAG 检索增强分析。")

    backend_mode = st.session_state.get("backend_runtime_mode", LOCAL_BACKEND_MODE)
    api_base_url = st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)
    selected_llm_provider = st.session_state.get(
        "selected_llm_provider",
        get_llm_provider_from_env(),
    )
    fallback_to_mock = st.session_state.get("fallback_to_mock", True)
    st.caption(f"当前运行后端模式：{backend_mode}")
    st.caption(f"当前 LLM Provider：{selected_llm_provider}")

    current_embedding_provider = get_embedding_provider()

    st.caption("RAG 模式会先从简历中检索与岗位最相关的片段，再交给大模型生成分析结果。")
    st.caption("RAG 模式支持本地 Embedding fallback。Gemini Embedding 免费 API 层级可能出现请求频率限制，如失败会自动使用本地向量。")
    st.caption(f"当前 RAG 向量模式：{current_embedding_provider}")
    if current_embedding_provider == "local_bge":
        st.caption("首次使用本地语义模型可能需要下载模型，耗时较长；下载完成后会使用本地缓存。")

    if st.session_state.get("demo_mode_enabled"):
        st.info("当前已加载示例数据。你仍然可以上传自己的简历文件，或直接编辑下方文本框内容。")

    st.subheader("Step 1：输入岗位 JD")
    job_description = st.text_area(
        "请输入岗位描述",
        height=220,
        placeholder="例如：粘贴 AI 应用开发实习生的岗位描述。",
        key="job_description_input",
    )

    st.subheader("Step 2：上传或填写简历")
    uploaded_file = st.file_uploader(
        "上传个人简历 / 经历文件",
        type=["txt", "pdf", "docx"],
        help="当前支持 txt、pdf、docx 文件。PDF 需为可复制文字的文本型 PDF。",
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

    resume_text_input = st.text_area(
        "请输入个人简历或项目经历",
        height=220,
        placeholder="如果没有上传文件，可以在这里粘贴你的专业技能、项目经历和自我评价内容。",
        key="resume_text_input",
    )

    final_resume_text = uploaded_resume_text.strip() if uploaded_resume_text.strip() else resume_text_input.strip()
    source_name = uploaded_source_name if uploaded_resume_text.strip() else "手动输入内容"

    st.subheader("Step 3：选择分析模式并生成报告")
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
                            use_mock_llm=selected_llm_provider == "mock",
                            fallback_to_mock=fallback_to_mock,
                        )
                    else:
                        result = run_agent_workflow(
                            job_description,
                            final_resume_text,
                            llm_provider=selected_llm_provider,
                            use_mock_llm=selected_llm_provider == "mock",
                            fallback_to_mock=fallback_to_mock,
                        )

            # 保存本次分析结果，避免点击下载按钮或切换 RAG 片段预览时页面 rerun 后结果消失。
            st.session_state["last_analysis_result"] = result
            st.session_state["last_analysis_rag_enabled"] = rag_enabled
            st.session_state["last_analysis_agent_workflow_enabled"] = agent_workflow_enabled

    result = st.session_state.get("last_analysis_result")
    result_rag_enabled = st.session_state.get("last_analysis_rag_enabled", False)
    result_agent_workflow_enabled = st.session_state.get("last_analysis_agent_workflow_enabled", False)

    if result:
        if result.get("error"):
            st.error(result["error"])
        else:
            st.success("分析完成")
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
                with st.expander("查看 Agent Workflow Steps", expanded=True):
                    for index, step in enumerate(result["workflow_steps"], start=1):
                        status = "成功" if step.get("success") else "失败"
                        st.markdown(f"**Step {index}: {step.get('step_name', '')}**")
                        st.write(f"tool_name：{step.get('tool_name', '')}")
                        st.write(f"success：{status}")
                        st.write(f"message：{step.get('message', '')}")
                        data_summary = step.get("data_summary") or {}
                        if data_summary:
                            st.json(data_summary)
                        if step.get("error"):
                            st.error(step["error"])

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
                with st.expander("查看 RAG 检索片段"):
                    show_full_rag_chunks = st.checkbox(
                        "显示完整 RAG 片段内容",
                        value=False,
                        key="show_full_rag_chunks",
                    )

                    for index, source in enumerate(rag_sources, start=1):
                        st.markdown(f"**片段 {index}**")
                        st.write(f"source：{source.get('source', 'resume')}")
                        st.write(f"file_name：{source.get('file_name', source.get('source_name', '未知来源'))}")
                        st.write(f"chunk_id：{source.get('chunk_id', source.get('chunk_index', index))}")
                        st.write(f"chunk_length：{source.get('chunk_length', len(source.get('text', '')))}")
                        st.write(f"section：{source.get('section', 'unknown')}")

                        if source.get("distance") is not None:
                            st.write(f"检索距离 distance：{source['distance']:.4f}（数值越小通常表示越相关）")
                        else:
                            st.write("distance：未返回")

                        if source.get("rerank_score") is not None:
                            st.write(f"rerank_score：{source['rerank_score']:.4f}")
                            st.write(f"keyword_hits：{source.get('keyword_hits', [])}")
                            st.write(f"section_bonus：{source.get('section_bonus', 0):.4f}")

                        chunk_text = source.get("text", "")
                        preview_text = chunk_text
                        if not show_full_rag_chunks and len(chunk_text) > 400:
                            preview_text = f"{chunk_text[:400]}……"

                        st.text_area(
                            f"片段内容预览 {index}",
                            value=preview_text,
                            height=140,
                            disabled=True,
                            key=f"rag_source_{index}",
                        )

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
    st.title("关于项目 / 后续计划")

    st.subheader("当前项目已完成")
    st.markdown("- 文件解析：txt / pdf / docx")
    st.markdown("- Prompt 分析：普通分析和 RAG Prompt")
    st.markdown("- RAG 检索：chunk、embedding、top_k 召回")
    st.markdown("- ChromaDB 本地向量库")
    st.markdown("- local embedding fallback")
    st.markdown("- 示例模式")
    st.markdown("- Markdown 报告导出")
    st.markdown("- 基础测试脚本 simple_test.py")
    st.markdown("- Tool Calling / Agent Workflow / Trace")
    st.markdown("- Lightweight Rerank / RAG Evaluation")
    st.markdown("- FastAPI 接口与 local / API 双模式")

    st.subheader("当前不足")
    st.markdown("- 当前更像 RAG Workflow，不是完整多工具 Agent。")
    st.markdown("- Gemini API 可能受地区和额度影响。")
    st.markdown("- local hash embedding 语义能力有限。")
    st.markdown("- Rerank 是规则方法，不是 cross-encoder。")
    st.markdown("- Eval case 和 gold evidence 数量仍较少。")
    st.markdown("- FastAPI 是接口 MVP，没有鉴权、数据库和任务队列。")
    st.markdown("- API mode 的 RAG 报告生成仍有本地 LLM 调用。")

    st.subheader("后续计划")
    st.markdown("- metadata 过滤")
    st.markdown("- cross-encoder rerank")
    st.markdown("- 完整 API 化 RAG 报告生成")
    st.markdown("- 更严格的检索与答案质量评测")
    st.markdown("- Trace 集中存储与查询")
    st.markdown("- DeepSeek / OpenAI-compatible 模型切换")
    st.markdown("- 更完整测试")


st.set_page_config(
    page_title="AI Resume Agent",
    page_icon="🤖",
    layout="wide",
)

with st.sidebar:
    st.header("功能入口")
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
    selected_llm_provider = st.selectbox(
        "选择模型调用方式",
        LLM_PROVIDER_OPTIONS,
        key="selected_llm_provider",
    )
    if selected_llm_provider == "openai_compatible":
        st.info(
            "需要在 .env 中配置 OPENAI_COMPATIBLE_API_KEY、"
            "OPENAI_COMPATIBLE_BASE_URL 和 OPENAI_COMPATIBLE_MODEL。"
        )
    elif selected_llm_provider == "mock":
        st.warning("当前使用 mock LLM，只用于测试流程，不代表真实模型质量。")

    if st.button("检查 LLM Provider", key="check_llm_provider"):
        if backend_mode == API_BACKEND_MODE:
            health_call = check_llm_health_api(
                api_base_url,
                provider=selected_llm_provider,
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
    st.caption("AI 应用工程化 MVP / RAG Workflow 原型")
    st.caption(f"Embedding Provider：{get_embedding_provider()}")

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
