import json
import os
from pathlib import Path

os.environ["EMBEDDING_PROVIDER"] = "local"

from agent_workflow import evaluate_retrieval_quality, run_resume_agent_workflow
from api_client import (
    call_agent_workflow_api,
    call_markdown_report_api,
    call_rag_retrieve_api,
    check_api_health,
)
from api_server import AgentWorkflowRequest, MarkdownReportRequest, RagRetrieveRequest, app as api_app
from eval_runner import discover_eval_cases, load_eval_case, run_evaluations
from llm_provider import (
    LLMResult,
    ProviderHealthResult,
    check_llm_provider_health,
    generate_with_llm,
    get_llm_provider_from_env,
)
from rag import build_chunk_records, detect_resume_sections, get_local_embedding, split_text
from rag_eval_utils import evaluate_retrieval_result
from rerank_utils import extract_keywords, rerank_chunks
from tools import ToolResult, rag_retrieve_tool, review_report_tool
from trace_utils import TraceStep, WorkflowTrace, create_run_id, now_iso, trace_to_dict


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


def test_tool_result_structure() -> None:
    result = ToolResult(
        success=True,
        tool_name="demo_tool",
        message="ok",
        data={"value": 1},
    )
    result_dict = result.to_dict()
    assert_true(result_dict["success"] is True, "ToolResult 应该包含 success")
    assert_true(result_dict["tool_name"] == "demo_tool", "ToolResult 应该包含 tool_name")
    assert_true(result_dict["error"] is None, "ToolResult 成功时 error 应该为 None")


def test_rag_retrieve_tool_basic() -> None:
    resume_text = """
姓名：测试同学
专业技能：Python、Streamlit、RAG、ChromaDB
项目经历：实现过简历和岗位匹配分析 Demo。
"""
    job_description = "招聘 AI 应用开发实习生，要求 Python、RAG、向量数据库经验。"
    result = rag_retrieve_tool(
        resume_text=resume_text,
        job_description=job_description,
        top_k=2,
    )
    assert_true(result.success, f"rag_retrieve_tool 不应直接崩溃：{result.error}")
    assert_true("chunks" in result.data, "rag_retrieve_tool 应该返回 chunks")
    assert_true(result.data["retrieved_chunk_count"] >= 1, "简单输入至少应召回 1 个 chunk")


def test_agent_workflow_steps_with_mock_llm() -> None:
    resume_text = "专业技能：Python、RAG、ChromaDB。项目经历：AI Resume Agent。"
    job_description = "AI 应用开发实习生，要求 Python 和 RAG 项目经验。"

    def mock_llm(_prompt: str) -> str:
        return """
## 岗位要求分析
需要 Python 和 RAG 项目经验。

## 个人能力分析
简历中体现了 Python、RAG 和 ChromaDB。

## 匹配度分析
候选人与岗位有基础匹配。

## 简历优化建议
补充量化指标和项目效果。
"""

    result = run_resume_agent_workflow(
        resume_text=resume_text,
        job_description=job_description,
        top_k=2,
        use_rag=True,
        llm_callable=mock_llm,
    )
    assert_true("workflow_steps" in result, "Agent Workflow 应该返回 workflow_steps")
    assert_true(len(result["workflow_steps"]) >= 2, "Agent Workflow 应该至少包含检索和分析两步")
    assert_true(all("tool_name" in step for step in result["workflow_steps"]), "每一步应该记录 tool_name")
    assert_true("trace" in result, "Agent Workflow 应该返回 trace")
    assert_true(bool(result["trace"].get("run_id")), "trace 应该包含 run_id")
    assert_true(len(result["trace"].get("steps", [])) >= 2, "trace 应该记录工具步骤")
    assert_true(result["trace"].get("duration_ms") is not None, "trace 应该记录总耗时")
    assert_true(bool(result["trace"].get("start_time")), "trace 应该记录开始时间")
    assert_true(bool(result["trace"].get("end_time")), "trace 应该记录结束时间")


def test_trace_structures_and_json_serialization() -> None:
    run_id = create_run_id()
    assert_true(isinstance(run_id, str) and bool(run_id), "create_run_id 应该生成非空字符串")

    timestamp = now_iso()
    step = TraceStep(
        step_name="demo_step",
        tool_name="demo_tool",
        success=True,
        message="ok",
        input_summary={"input_length": 10},
        output_summary={"result_count": 1},
        start_time=timestamp,
        end_time=timestamp,
        duration_ms=1.5,
    )
    trace = WorkflowTrace(
        run_id=run_id,
        mode="test",
        start_time=timestamp,
        end_time=timestamp,
        duration_ms=1.5,
        resume_length=10,
        job_description_length=20,
        top_k=2,
        embedding_provider="local",
        used_rag=True,
        used_fallback=False,
        steps=[step],
        final_status="success",
    )
    trace_dict = trace_to_dict(trace)
    serialized = json.dumps(trace_dict, ensure_ascii=False)
    assert_true(bool(serialized), "trace_to_dict 结果应该可以被 json.dumps 序列化")
    assert_true(trace_dict["steps"][0]["tool_name"] == "demo_tool", "trace 应该保留步骤信息")


def test_agent_workflow_trace_without_rag() -> None:
    def mock_llm(_prompt: str) -> str:
        return "## 岗位要求分析\n测试结果"

    result = run_resume_agent_workflow(
        resume_text="Python 项目经历",
        job_description="Python 开发岗位",
        use_rag=False,
        llm_callable=mock_llm,
    )
    trace = result["trace"]
    assert_true(trace["used_rag"] is False, "use_rag=False 时 trace 应该记录未使用 RAG")
    assert_true(trace["steps"][0]["output_summary"]["skipped"] is True, "trace 应该记录 RAG 已跳过")


def test_eval_cases_and_runner_imports() -> None:
    eval_cases_dir = BASE_DIR / "eval_cases"
    assert_true(eval_cases_dir.is_dir(), "eval_cases 目录应该存在")

    case_paths = discover_eval_cases(eval_cases_dir)
    assert_true(len(case_paths) >= 1, "eval_cases 至少应该包含一个完整 case")
    for case_path in case_paths:
        for file_name in ["resume.txt", "jd.txt", "expected.json"]:
            assert_true((case_path / file_name).is_file(), f"{case_path.name} 缺少 {file_name}")
        loaded_case = load_eval_case(case_path)
        assert_true(bool(loaded_case["resume_text"]), f"{case_path.name} resume.txt 不应该为空")
        assert_true(bool(loaded_case["job_description"]), f"{case_path.name} jd.txt 不应该为空")
        assert_true(
            bool(loaded_case["expected"].get("gold_evidence")),
            f"{case_path.name} expected.json 应该包含 gold_evidence",
        )

    assert_true(callable(run_evaluations), "eval_runner.run_evaluations 应该可以被 import")


def test_demo_polish_files_and_readme_sections() -> None:
    assert_true((BASE_DIR / ".env.example").is_file(), ".env.example 应该存在")
    assert_true(
        (BASE_DIR / "examples" / "example_report.md").is_file(),
        "examples/example_report.md 应该存在",
    )

    readme_text = (BASE_DIR / "README.md").read_text(encoding="utf-8")
    for section_name in ["Agent Workflow", "Trace", "Eval Runner", "项目边界"]:
        assert_true(section_name in readme_text, f"README.md 应该包含 {section_name} 小节")


def test_extract_keywords_and_rerank_chunks() -> None:
    keywords = extract_keywords("需要 Python、RAG、Agent 和 FastAPI 项目经验")
    assert_true("Python" in keywords, "extract_keywords 应该识别 Python")
    assert_true("RAG" in keywords, "extract_keywords 应该识别 RAG")
    assert_true("Agent" in keywords, "extract_keywords 应该识别 Agent")

    chunks = [
        {"text": "负责日常文档整理", "section": "basic_info", "distance": 0.1},
        {"text": "使用 Python 和 RAG 构建 Agent 项目", "section": "project_experience", "distance": 0.4},
    ]
    reranked = rerank_chunks(chunks, "Python RAG Agent 开发", top_k=2)
    assert_true(reranked[0]["section"] == "project_experience", "rerank 应优先关键词匹配片段")
    assert_true("rerank_score" in reranked[0], "rerank 结果应该包含 rerank_score")
    assert_true(bool(reranked[0]["keyword_hits"]), "rerank 结果应该包含 keyword_hits")


def test_rag_tool_and_agent_workflow_with_rerank() -> None:
    resume_text = """专业技能
Python、RAG、ChromaDB、Agent
项目经历
使用 Python 实现 RAG 简历分析工具，并展示召回片段。"""
    job_description = "招聘 Python RAG Agent 开发实习生"
    retrieval = rag_retrieve_tool(
        resume_text=resume_text,
        job_description=job_description,
        top_k=2,
        use_rerank=True,
    )
    assert_true(retrieval.success, f"rerank RAG 工具不应该失败：{retrieval.error}")
    assert_true(retrieval.data["used_rerank"] is True, "工具结果应该记录 used_rerank")
    assert_true(
        all("rerank_score" in chunk for chunk in retrieval.data["chunks"]),
        "rerank chunks 应该包含 rerank_score",
    )

    def mock_llm(_prompt: str) -> str:
        return "## 岗位要求分析\nPython RAG Agent\n## 个人能力分析\n匹配"

    workflow = run_resume_agent_workflow(
        resume_text=resume_text,
        job_description=job_description,
        top_k=2,
        use_rag=True,
        use_rerank=True,
        llm_callable=mock_llm,
    )
    assert_true(workflow["success"], "启用 rerank 的 Agent Workflow 应该成功")
    assert_true(workflow["trace"]["used_rerank"] is True, "Trace 应该记录 used_rerank")
    assert_true(
        workflow["trace"]["rerank_method"] == "rule_based",
        "Trace 应该记录 rule_based rerank 方法",
    )
    assert_true(
        workflow["trace"]["steps"][0]["output_summary"]["rerank_score"] is not None,
        "Trace 步骤应该记录 rerank_score",
    )


def test_fastapi_files_and_request_models() -> None:
    assert_true((BASE_DIR / "api_server.py").is_file(), "api_server.py 应该存在")
    assert_true((BASE_DIR / "run_api.bat").is_file(), "run_api.bat 应该存在")
    assert_true(api_app.title == "AI Resume Agent API", "FastAPI app 应该可以 import")

    rag_request = RagRetrieveRequest(
        resume_text="Python RAG 项目",
        job_description="AI 应用开发岗位",
    )
    agent_request = AgentWorkflowRequest(
        resume_text="Python Agent 项目",
        job_description="Agent 开发岗位",
    )
    report_request = MarkdownReportRequest(
        analysis="分析结果",
        metadata={"mode": "agent_workflow"},
    )
    assert_true(rag_request.top_k == 5, "RAG 请求 top_k 默认值应该为 5")
    assert_true(agent_request.use_rag is True, "Agent 请求默认应该启用 RAG")
    assert_true(agent_request.use_rerank is False, "Agent 请求默认不启用 rerank")
    assert_true(report_request.metadata["mode"] == "agent_workflow", "Markdown metadata 应可创建")


def test_final_hardening_documents() -> None:
    document_paths = [
        "FINAL_CHECKLIST.md",
        "INTERVIEW_NOTES.md",
        "docs/architecture.md",
        "docs/api.md",
        "docs/eval.md",
    ]
    readme_text = (BASE_DIR / "README.md").read_text(encoding="utf-8")
    for relative_path in document_paths:
        assert_true((BASE_DIR / relative_path).is_file(), f"{relative_path} 应该存在")
        assert_true(relative_path in readme_text, f"README.md 应该链接 {relative_path}")


def test_rag_evaluation_metrics() -> None:
    expected = {
        "expected_sections": ["skills", "project_experience"],
        "expected_keywords": ["Python", "RAG", "Agent"],
        "gold_evidence": [
            {"section": "skills", "keywords": ["Python", "RAG"]},
            {"section": "project_experience", "keywords": ["Agent"]},
        ],
    }
    empty_metrics = evaluate_retrieval_result([], expected)
    assert_true(empty_metrics["recall_at_k"]["3"] == 0.0, "空召回 Recall@3 应为 0")
    assert_true(empty_metrics["mrr"] == 0.0, "空召回 MRR 应为 0")

    chunks = [
        {"text": "个人基本信息", "section": "basic_info"},
        {"content": "Python 与 RAG 技能", "section": "skills"},
        {"document": "Agent 项目经验", "metadata": {"section": "project_experience"}},
    ]
    metrics = evaluate_retrieval_result(chunks, expected)
    assert_true("recall_at_k" in metrics, "RAG 指标应该包含 recall_at_k")
    assert_true(metrics["recall_at_k"]["1"] == 0.0, "第一条无关时 Recall@1 应为 0")
    assert_true(metrics["recall_at_k"]["3"] == 1.0, "前三条应命中全部 gold evidence")
    assert_true(metrics["mrr"] == 0.5, "首个 evidence 在 rank 2 时 MRR 应为 0.5")


def test_api_client_imports_and_offline_error() -> None:
    for api_function in [
        check_api_health,
        call_rag_retrieve_api,
        call_agent_workflow_api,
        call_markdown_report_api,
    ]:
        assert_true(callable(api_function), "api_client 函数应该可以 import")

    offline_result = check_api_health("http://127.0.0.1:1")
    assert_true(offline_result["success"] is False, "无效 API 地址应该返回 success=False")
    assert_true(bool(offline_result["error"]), "无效 API 地址应该返回友好错误")


def test_multi_model_provider_offline_paths() -> None:
    # --- basic data structure creation ---
    created = LLMResult(
        success=True,
        provider="mock",
        model="test-model",
        text="测试",
    )
    assert_true(created.provider == "mock", "LLMResult 应该可以创建")

    # --- explicit mock provider ---
    mock_result = generate_with_llm("测试 Prompt", use_mock=True)
    assert_true(mock_result.success, "mock LLM 应该返回 success=True")
    assert_true(mock_result.provider == "mock", "mock LLM 应记录 provider=mock")
    assert_true("岗位要求分析" in mock_result.text, "mock LLM 应返回稳定结构")

    # --- mock health check ---
    health_record = ProviderHealthResult(
        provider="mock",
        model="mock-structured-v1",
        available=True,
        message="ok",
    )
    assert_true(health_record.available, "ProviderHealthResult 应该可以创建")
    mock_health = check_llm_provider_health(provider="mock")
    assert_true(mock_health.available, "Mock Provider 健康检查应该通过")

    # --- get_llm_provider_from_env() must NOT silently downgrade on missing key ---
    original_gemini_key = os.environ.pop("GEMINI_API_KEY", None)
    original_llm_prov = os.environ.get("LLM_PROVIDER")
    try:
        os.environ["LLM_PROVIDER"] = "gemini"
        resolved = get_llm_provider_from_env()
        assert_true(
            resolved == "gemini",
            f"缺少 GEMINI_API_KEY 时 get_llm_provider_from_env() 应返回 'gemini'，实际返回 {resolved}",
        )
    finally:
        if original_gemini_key is not None:
            os.environ["GEMINI_API_KEY"] = original_gemini_key
        if original_llm_prov is not None:
            os.environ["LLM_PROVIDER"] = original_llm_prov
        else:
            os.environ.pop("LLM_PROVIDER", None)

    # --- gemini fallback when GEMINI_API_KEY is missing ---
    original_gemini_key = os.environ.pop("GEMINI_API_KEY", None)
    try:
        gemini_no_key = generate_with_llm(
            "测试 Prompt",
            provider="gemini",
            fallback_to_mock=False,
        )
        gemini_fallback = generate_with_llm(
            "测试 Prompt",
            provider="gemini",
            fallback_to_mock=True,
        )
    finally:
        if original_gemini_key is not None:
            os.environ["GEMINI_API_KEY"] = original_gemini_key

    assert_true(gemini_no_key.success is False, "缺少 Gemini Key 且不 fallback 时应失败")
    assert_true("API Key" in (gemini_no_key.error or ""), "错误应指出缺少 API Key")
    assert_true(gemini_fallback.success, "启用 fallback 后缺少 Gemini Key 应返回 Mock 结果")
    assert_true(gemini_fallback.provider == "mock", "fallback provider 应该为 mock")
    assert_true(gemini_fallback.fallback_used, "Gemini fallback 结果应记录 fallback_used=True")
    assert_true(
        gemini_fallback.original_provider == "gemini",
        "Gemini fallback 结果应记录 original_provider=gemini",
    )
    assert_true(
        bool(gemini_fallback.error),
        "Gemini fallback 结果应保留 provider_error",
    )

    # --- deepseek fallback when DEEPSEEK_API_KEY is missing ---
    original_ds_key = os.environ.pop("DEEPSEEK_API_KEY", None)
    original_llm_provider = os.environ.get("LLM_PROVIDER")
    try:
        # get_llm_provider_from_env should return "deepseek" when configured
        os.environ["LLM_PROVIDER"] = "deepseek"
        resolved_ds = get_llm_provider_from_env()
        assert_true(
            resolved_ds == "deepseek",
            f"LLM_PROVIDER=deepseek 时应返回 'deepseek'，实际返回 {resolved_ds}",
        )

        ds_no_key = generate_with_llm(
            "测试 Prompt",
            provider="deepseek",
            fallback_to_mock=False,
        )
        ds_fallback = generate_with_llm(
            "测试 Prompt",
            provider="deepseek",
            fallback_to_mock=True,
        )
    finally:
        if original_ds_key is not None:
            os.environ["DEEPSEEK_API_KEY"] = original_ds_key
        else:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        if original_llm_provider is not None:
            os.environ["LLM_PROVIDER"] = original_llm_provider
        else:
            os.environ.pop("LLM_PROVIDER", None)

    assert_true(ds_no_key.success is False, "缺少 DeepSeek Key 且不 fallback 时应失败")
    assert_true("API Key" in (ds_no_key.error or ""), "错误应指出缺少 API Key")
    assert_true(ds_fallback.success, "启用 fallback 后缺少 DeepSeek Key 应返回 Mock 结果")
    assert_true(ds_fallback.provider == "mock", "fallback provider 应该为 mock")
    assert_true(ds_fallback.fallback_used, "DeepSeek fallback 结果应记录 fallback_used=True")
    assert_true(
        ds_fallback.original_provider == "deepseek",
        "DeepSeek fallback 结果应记录 original_provider=deepseek",
    )
    assert_true(
        bool(ds_fallback.error),
        "DeepSeek fallback 结果应保留 provider_error",
    )

    # --- Agent Workflow with provider="deepseek" must not report "unsupported" ---
    original_ds_key2 = os.environ.pop("DEEPSEEK_API_KEY", None)
    original_llm_provider2 = os.environ.get("LLM_PROVIDER")
    try:
        os.environ["LLM_PROVIDER"] = "deepseek"
        ds_workflow = run_resume_agent_workflow(
            resume_text="Python RAG 项目",
            job_description="AI 应用开发岗位",
            use_rag=False,
            llm_provider="deepseek",
            fallback_to_mock=True,
        )
    finally:
        if original_ds_key2 is not None:
            os.environ["DEEPSEEK_API_KEY"] = original_ds_key2
        else:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        if original_llm_provider2 is not None:
            os.environ["LLM_PROVIDER"] = original_llm_provider2
        else:
            os.environ.pop("LLM_PROVIDER", None)

    assert_true(ds_workflow["success"], "Agent Workflow with deepseek+fallback 应该成功")
    assert_true(ds_workflow["trace"]["fallback_used"] is True, "Trace 应记录 fallback_used=True")
    assert_true(
        ds_workflow["trace"]["original_provider"] == "deepseek",
        "Trace 应记录 original_provider=deepseek",
    )
    assert_true(
        ds_workflow["trace"]["llm_provider"] == "mock",
        "Fallback 后 llm_provider 应为 mock",
    )
    err_msg = ds_workflow.get("error") or ""
    assert_true(
        "不支持的" not in err_msg,
        f"不应报不支持的 provider 错误，实际 error: {err_msg}",
    )

    # --- openai_compatible fallback when keys are missing ---
    original_key = os.environ.pop("OPENAI_COMPATIBLE_API_KEY", None)
    original_url = os.environ.pop("OPENAI_COMPATIBLE_BASE_URL", None)
    try:
        missing_health = check_llm_provider_health(provider="openai_compatible")
        missing_key_result = generate_with_llm(
            "测试 Prompt",
            provider="openai_compatible",
            fallback_to_mock=False,
        )
        fallback_result = generate_with_llm(
            "测试 Prompt",
            provider="openai_compatible",
            fallback_to_mock=True,
        )
        fallback_workflow = run_resume_agent_workflow(
            resume_text="Python RAG 项目",
            job_description="AI 应用开发岗位",
            use_rag=False,
            llm_provider="openai_compatible",
            fallback_to_mock=True,
        )
    finally:
        if original_key is not None:
            os.environ["OPENAI_COMPATIBLE_API_KEY"] = original_key
        if original_url is not None:
            os.environ["OPENAI_COMPATIBLE_BASE_URL"] = original_url
    assert_true(missing_health.available is False, "缺少兼容配置时健康检查应失败")
    assert_true(missing_key_result.success is False, "缺少兼容 API Key 时应该友好失败")
    assert_true("API Key" in missing_key_result.error, "错误应指出缺少 API Key")
    assert_true(fallback_result.success, "启用 fallback 后应该返回 Mock 结果")
    assert_true(fallback_result.provider == "mock", "fallback provider 应该为 mock")
    assert_true(fallback_result.fallback_used, "fallback 结果应该记录 fallback_used")
    assert_true(
        fallback_result.original_provider == "openai_compatible",
        "fallback 结果应该记录 original_provider",
    )
    assert_true(fallback_workflow["success"], "fallback Agent Workflow 应该成功")
    assert_true(fallback_workflow["trace"]["fallback_used"] is True, "Trace 应记录 fallback_used")
    assert_true(
        fallback_workflow["trace"]["original_provider"] == "openai_compatible",
        "Trace 应记录 original_provider",
    )
    assert_true(bool(fallback_workflow["trace"]["provider_error"]), "Trace 应记录 provider_error")

    # --- mock provider through Agent Workflow ---
    workflow = run_resume_agent_workflow(
        resume_text="Python RAG 项目",
        job_description="AI 应用开发岗位",
        use_rag=False,
        llm_provider="mock",
        use_mock_llm=True,
    )
    assert_true(workflow["success"], "Agent Workflow mock provider 路径应该成功")
    assert_true(workflow["trace"]["llm_provider"] == "mock", "Trace 应记录 mock provider")
    assert_true(workflow["trace"]["use_mock_llm"] is True, "Trace 应记录 use_mock_llm")


def test_lightweight_agent_harness() -> None:
    """Verify reviewer tool, retrieval quality, and agent workflow harness integration."""

    # --- review_report_tool: empty analysis ---
    empty_review = review_report_tool(
        job_description="Python 开发岗位",
        retrieved_chunks=[],
        analysis_text="",
    )
    assert_true(empty_review.success, "review tool 自身应成功运行")
    assert_true(empty_review.data["review_passed"] is False, "空 analysis 应 review_passed=False")
    assert_true("分析文本为空" in empty_review.data["missing_points"], "应提示分析文本为空")

    # --- review_report_tool: missing structures ---
    partial_analysis = "## 岗位要求分析\n需要 Python。\n## 个人能力分析\n会 Python。"
    partial_review = review_report_tool(
        job_description="Python 开发岗位",
        retrieved_chunks=[{"text": "Python", "section": "skills"}],
        analysis_text=partial_analysis,
    )
    assert_true(partial_review.data["review_passed"] is False, "缺少匹配/优化结构应不通过")
    assert_true(len(partial_review.data["missing_points"]) > 0, "应该有缺失结构列表")

    # --- review_report_tool: full structure ---
    full_analysis = """## 岗位要求分析
需要 Python 开发能力。

## 个人能力分析
候选人具备 Python 项目经验。

## 匹配度分析
岗位要求与候选人技能匹配度较高。

## 简历优化建议
建议补充量化指标和项目效果。
"""
    full_review = review_report_tool(
        job_description="Python RAG 开发岗位",
        retrieved_chunks=[{"text": "Python RAG 项目", "section": "project_experience"}],
        analysis_text=full_analysis,
    )
    assert_true(full_review.data["review_passed"] is True, "完整结构应 review_passed=True")
    assert_true("项目" in full_review.data["evidence_usage"], "应检测到项目证据")

    # --- evaluate_retrieval_quality ---
    empty_quality, empty_reason = evaluate_retrieval_quality([])
    assert_true(empty_quality == "low", "空 chunks 应返回 low")

    ok_quality, ok_reason = evaluate_retrieval_quality([
        {"keyword_hits": ["Python"], "rerank_score": 0.8},
    ])
    assert_true(ok_quality == "ok", "有关键词命中应返回 ok")

    # --- Agent Workflow returns review_result ---
    def mock_llm(prompt: str) -> str:
        return full_analysis

    result = run_resume_agent_workflow(
        resume_text="Python RAG 项目经历",
        job_description="Python RAG 开发岗位",
        top_k=2,
        use_rag=True,
        llm_callable=mock_llm,
    )
    assert_true(result["success"], "Agent Workflow should succeed")
    review = result.get("review_result") or {}
    assert_true(review.get("review_passed") is True, "Workflow 应返回 review_passed")
    # Refinement may or may not trigger depending on local embedding — only check bounding
    attempts = result.get("retrieval_attempts", 1)
    assert_true(
        1 <= attempts <= 2,
        f"retrieval_attempts 应在 [1, 2] 范围内，实际 {attempts}",
    )
    if result.get("query_refinement_used"):
        assert_true(attempts == 2, "refinement 触发时 attempts 应为 2")

    # --- query refinement is bounded (no infinite loop) ---
    # Small resume text with few chunks ensures quality=low → triggers refinement
    tiny_result = run_resume_agent_workflow(
        resume_text="Python",
        job_description="Python RAG Agent 开发",
        top_k=2,
        use_rag=True,
        llm_callable=mock_llm,
    )
    attempts = tiny_result.get("retrieval_attempts", 1)
    assert_true(attempts <= 2, f"retrieval_attempts 不应超过 2，实际 {attempts}")


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
    test_tool_result_structure()
    test_rag_retrieve_tool_basic()
    test_agent_workflow_steps_with_mock_llm()
    test_trace_structures_and_json_serialization()
    test_agent_workflow_trace_without_rag()
    test_eval_cases_and_runner_imports()
    test_demo_polish_files_and_readme_sections()
    test_extract_keywords_and_rerank_chunks()
    test_rag_tool_and_agent_workflow_with_rerank()
    test_fastapi_files_and_request_models()
    test_final_hardening_documents()
    test_rag_evaluation_metrics()
    test_api_client_imports_and_offline_error()
    test_multi_model_provider_offline_paths()
    test_lightweight_agent_harness()
    print("simple_test.py: all tests passed")
