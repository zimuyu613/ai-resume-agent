from pathlib import Path

from rag import build_chunk_records, detect_resume_sections, get_local_embedding, split_text


BASE_DIR = Path(__file__).parent


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_split_text() -> None:
    text = "这是第一段简历内容。" * 100
    chunks = split_text(text, chunk_size=120, overlap=20)
    assert_true(len(chunks) > 1, "split_text 应该能把长文本切成多个 chunk")
    assert_true(all(chunk.strip() for chunk in chunks), "split_text 不应该产生空 chunk")


def test_chunk_metadata() -> None:
    text = "Python RAG 项目经历。" * 80
    records = build_chunk_records(text, chunk_size=120, overlap=20)
    assert_true(len(records) > 1, "build_chunk_records 应该能返回多个 chunk record")
    assert_true(all(record["text"].strip() for record in records), "chunk record 的 text 不应该为空")
    assert_true("chunk_id" in records[0], "chunk metadata 应该包含 chunk_id")
    assert_true("chunk_length" in records[0], "chunk metadata 应该包含 chunk_length")


def test_section_detection() -> None:
    text = """
姓名：张同学
求职意向：AI 应用开发实习生

教育背景
某某大学 计算机科学与技术

专业技能
Python、RAG、Prompt Engineering

项目经历
AI 简历匹配分析助手
    """
    sections = detect_resume_sections(text)
    section_names = {item["section"] for item in sections}
    assert_true("basic_info" in section_names, "应该能识别姓名和求职意向为 basic_info")
    assert_true("education" in section_names, "应该能识别教育背景")
    assert_true("skills" in section_names, "应该能识别专业技能")
    assert_true("project_experience" in section_names, "应该能识别项目经历")

    records = build_chunk_records(text, chunk_size=60, overlap=10)
    record_sections = {record["section"] for record in records}
    assert_true(record_sections != {"unknown"}, "chunk metadata 中 section 不应该总是 unknown")


def test_basic_info_section() -> None:
    text = """
姓名：李同学
邮箱：demo@example.com
手机：13800000000
求职意向：Python 开发实习生

专业技能
Python、Streamlit、ChromaDB、Prompt Engineering、RAG、向量数据库、文本切分、local embedding、Gemini API 调用。
熟悉使用 Python 构建 AI 应用原型，能够处理 txt、pdf、docx 文档解析和基础异常提示。

项目经历
RAG 简历分析助手：负责文本切分、向量检索、ChromaDB 入库、Prompt 拼接和分析结果展示。
项目中实现了 RAG 召回片段预览、section metadata、top_k 调整和 Markdown 报告导出。
"""
    records = build_chunk_records(text, chunk_size=80, overlap=10)
    sections = {record["section"] for record in records}
    assert_true("basic_info" in sections, "包含姓名和求职意向的文本应该识别出 basic_info")
    assert_true("skills" in sections, "包含专业技能的文本应该识别出 skills")
    assert_true("project_experience" in sections, "包含项目经历的文本应该识别出 project_experience")


def test_section_aware_chunking() -> None:
    text = """
姓名：王同学
求职意向：AI 应用开发实习生

教育背景
某某大学 计算机科学与技术

专业技能
Python、Streamlit、ChromaDB、Prompt Engineering、RAG、向量数据库、文本切分、local embedding、Gemini API 调用。
熟悉使用 Python 构建 AI 应用原型，能够处理 txt、pdf、docx 文档解析和基础异常提示。

项目经历
AI 简历匹配分析助手：负责文本切分、向量检索、ChromaDB 入库、Prompt 拼接和分析结果展示。
项目中实现了 RAG 召回片段预览、section metadata、top_k 调整和 Markdown 报告导出。
"""
    records = build_chunk_records(text, chunk_size=120, overlap=20)
    sections = {record["section"] for record in records}
    assert_true(sections != {"unknown"}, "section-aware chunking 后 section 不应全部是 unknown")

    skills_chunks = [record for record in records if record["section"] == "skills"]
    project_chunks = [record for record in records if record["section"] == "project_experience"]

    assert_true(skills_chunks, "应该生成 skills section 的 chunk")
    assert_true(project_chunks, "应该生成 project_experience section 的 chunk")
    assert_true(
        any("Python" in record["text"] or "ChromaDB" in record["text"] for record in skills_chunks),
        "skills chunk 内容应包含技能相关文本",
    )
    assert_true(
        any("AI 简历匹配分析助手" in record["text"] or "RAG 召回片段" in record["text"] for record in project_chunks),
        "project_experience chunk 内容应包含项目相关文本",
    )
    assert_true(
        all(not record["text"].lstrip().startswith(("姓名", "求职意向", "教育背景")) for record in project_chunks),
        "project_experience chunk 不应从姓名、求职意向或教育背景开头",
    )


def test_sample_resume_sections() -> None:
    resume_path = BASE_DIR / "samples" / "sample_resume.txt"
    resume_text = resume_path.read_text(encoding="utf-8")
    records = build_chunk_records(resume_text)
    sections = {record["section"] for record in records}
    assert_true(
        "skills" in sections or "project_experience" in sections,
        "sample_resume 至少应该识别出 skills 或 project_experience",
    )


def test_empty_text() -> None:
    chunks = split_text("")
    assert_true(chunks == [], "空文本应该返回空列表")


def test_local_embedding() -> None:
    embedding = get_local_embedding("Python RAG Prompt Engineering")
    assert_true(len(embedding) == 384, "local embedding 默认维度应该是 384")
    assert_true(any(value != 0 for value in embedding), "local embedding 不应该全为 0")


def test_sample_files() -> None:
    resume_path = BASE_DIR / "samples" / "sample_resume.txt"
    job_path = BASE_DIR / "samples" / "sample_job_description.txt"

    assert_true(resume_path.exists(), "sample_resume.txt 应该存在")
    assert_true(job_path.exists(), "sample_job_description.txt 应该存在")
    assert_true(resume_path.read_text(encoding="utf-8").strip(), "sample_resume.txt 不应该为空")
    assert_true(job_path.read_text(encoding="utf-8").strip(), "sample_job_description.txt 不应该为空")


if __name__ == "__main__":
    test_split_text()
    test_chunk_metadata()
    test_section_detection()
    test_basic_info_section()
    test_section_aware_chunking()
    test_sample_resume_sections()
    test_empty_text()
    test_local_embedding()
    test_sample_files()
    print("simple_test.py: all tests passed")
