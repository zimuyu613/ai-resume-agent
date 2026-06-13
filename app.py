from io import BytesIO
from pathlib import Path

import streamlit as st
from docx import Document
from pypdf import PdfReader

from agent import run_agent_workflow, run_rag_workflow
from rag import get_embedding_provider


BASE_DIR = Path(__file__).parent
SAMPLES_DIR = BASE_DIR / "samples"
SAMPLE_RESUME_PATH = SAMPLES_DIR / "sample_resume.txt"
SAMPLE_JOB_PATH = SAMPLES_DIR / "sample_job_description.txt"


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
        sections.extend(
            [
                "",
                "## RAG 检索说明",
                "",
                f"本次 RAG 检索召回 top_k = {top_k} 个片段。",
                f"- 本次简历生成 chunk 数量：{total_chunks}",
                f"- 用户设置 top_k：{top_k}",
                f"- 实际召回片段数量：{actual_count}",
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
                    f"- distance：{distance_text}",
                    "",
                    f"内容摘要：{preview}",
                    "",
                ]
            )

    return "\n".join(sections).strip() + "\n"


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
    rag_enabled = st.checkbox(
        "启用 RAG 检索增强分析",
        value=False,
        help="RAG 模式会先从简历中检索相关片段，再生成分析结果。",
    )
    rag_top_k = 3
    if rag_enabled:
        rag_top_k = st.slider("RAG 召回片段数量 top_k", min_value=1, max_value=8, value=3)
        st.caption("top_k 表示从向量数据库中召回最相关的前 k 个简历片段。")

    if st.button("开始分析"):
        if not job_description.strip() or not final_resume_text:
            st.warning("请先输入岗位描述，并上传或填写个人经历。")
        else:
            with st.spinner("正在分析中，请稍等..."):
                if rag_enabled:
                    result = run_rag_workflow(
                        job_description,
                        final_resume_text,
                        source_name=source_name,
                        top_k=rag_top_k,
                    )
                else:
                    result = run_agent_workflow(job_description, final_resume_text)

            # 保存本次分析结果，避免点击下载按钮或切换 RAG 片段预览时页面 rerun 后结果消失。
            st.session_state["last_analysis_result"] = result
            st.session_state["last_analysis_rag_enabled"] = rag_enabled

    result = st.session_state.get("last_analysis_result")
    result_rag_enabled = st.session_state.get("last_analysis_rag_enabled", False)

    if result:
        if result.get("error"):
            st.error(result["error"])
        else:
            st.success("分析完成")
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

                        if source.get("distance") is not None:
                            st.write(f"检索距离 distance：{source['distance']:.4f}（数值越小通常表示越相关）")
                        else:
                            st.write("distance：未返回")

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

    st.subheader("当前不足")
    st.markdown("- 当前更像 RAG Workflow，不是完整多工具 Agent。")
    st.markdown("- Gemini API 可能受地区和额度影响。")
    st.markdown("- local hash embedding 语义能力有限。")
    st.markdown("- 还没有 rerank。")
    st.markdown("- 还没有系统化检索质量评估。")
    st.markdown("- 还没有 FastAPI 服务化。")
    st.markdown("- 还没有完整 Trace / 日志面板。")

    st.subheader("后续计划")
    st.markdown("- metadata 过滤")
    st.markdown("- rerank")
    st.markdown("- FastAPI 服务化")
    st.markdown("- Tool Calling")
    st.markdown("- Trace 执行轨迹")
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
