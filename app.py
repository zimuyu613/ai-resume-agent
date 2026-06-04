import streamlit as st
from io import BytesIO
from pypdf import PdfReader
from docx import Document

from agent import run_agent_workflow


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


st.set_page_config(
    page_title="AI Agent 简历与岗位匹配分析助手",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Agent 简历与岗位匹配分析助手")

st.write(
    "输入岗位描述和个人经历，系统会分析岗位要求、个人匹配点、能力差距和简历优化建议。"
)

st.info(
    "当前版本支持手动输入个人经历，也支持上传 txt、pdf、docx 简历文件。"
    "建议输入内容保持精简，以保证分析稳定性。"
)

with st.sidebar:
    st.header("项目说明")
    st.write("这是一个用于学习 AI Agent Workflow 的小型原型项目。")
    st.write("当前版本包含：")
    st.write("- 岗位要求分析")
    st.write("- 个人能力分析")
    st.write("- 匹配度分析")
    st.write("- 简历优化建议")
    st.write("- txt / pdf / docx 简历文件上传")
    st.write("")
    st.write("后续可扩展：")
    st.write("- RAG 文档检索")
    st.write("- Embedding 向量化")
    st.write("- ChromaDB / FAISS 向量数据库")
    st.write("- Tool Calling")
    st.write("- FastAPI 后端服务化")

job_description = st.text_area(
    "请输入岗位描述",
    height=240,
    placeholder="例如：粘贴 AI Agent 开发实习生的岗位描述。",
)

uploaded_file = st.file_uploader(
    "上传个人简历 / 经历文件",
    type=["txt", "pdf", "docx"],
    help="当前支持 txt、pdf、docx 文件。PDF 需为可复制文字的文本型 PDF。",
)

uploaded_resume_text = ""

if uploaded_file is not None:
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
    height=240,
    placeholder="如果没有上传文件，可以在这里粘贴你的专业技能、项目经历和自我评价内容。",
)

# 优先使用上传文件内容；如果没有上传文件，则使用手动输入内容
final_resume_text = uploaded_resume_text.strip() if uploaded_resume_text.strip() else resume_text_input.strip()

if st.button("开始分析"):
    if not job_description.strip() or not final_resume_text:
        st.warning("请先输入岗位描述，并上传或填写个人经历。")
    else:
        with st.spinner("正在分析中，请稍等..."):
            result = run_agent_workflow(job_description, final_resume_text)

        st.success("分析完成")

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