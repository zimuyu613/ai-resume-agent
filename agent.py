import os
import time
from dotenv import load_dotenv
from google import genai

from prompts import (
    SYSTEM_PROMPT,
    COMPREHENSIVE_ANALYSIS_PROMPT,
    RAG_ANALYSIS_PROMPT,
)
from rag import retrieve_relevant_chunks_with_sources

# 读取 .env 文件中的环境变量
load_dotenv()

# 初始化 Gemini 客户端
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 从 .env 读取模型名称，如果没有配置，则使用默认模型
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def format_llm_error(error: Exception) -> str:
    """
    将 Gemini 常见错误转换成面向用户的简洁提示。
    详细错误保留在控制台，方便开发调试。
    """
    error_msg = str(error)
    error_lower = error_msg.lower()
    print(f"Gemini 调用详细错误：{error_msg}")

    if "429" in error_msg or "resource_exhausted" in error_lower or "quota" in error_lower:
        return (
            "调用模型时出现错误：Gemini API 额度不足或请求过多。"
            "请稍后重试、降低请求频率，或更换可用的 API Key / 计费方案。"
        )

    if "user location is not supported" in error_lower or "location is not supported" in error_lower:
        return (
            "调用模型时出现错误：当前地区可能不支持 Gemini API。"
            "请检查网络环境，或切换到可用的模型服务。"
        )

    model_error_keywords = [
        "model name",
        "model not found",
        "unexpected model name format",
        "invalid argument",
    ]
    if any(keyword in error_lower for keyword in model_error_keywords):
        return (
            "调用模型时出现错误：模型名称配置可能不正确。"
            "请检查 .env 中的 GEMINI_MODEL，例如 gemini-2.5-flash 或 gemini-2.0-flash。"
        )

    if "api key" in error_lower or "permission_denied" in error_lower or "unauthenticated" in error_lower:
        return "调用模型时出现错误：API Key 无效或缺少权限，请检查 .env 中的 GEMINI_API_KEY。"

    return "调用模型时出现错误：模型服务暂时不可用，请稍后重试或检查控制台日志。"


def call_llm(prompt: str) -> str:
    """
    调用 Gemini 大语言模型并返回文本结果。
    增加简单重试机制，用于处理 503 高负载等临时错误。
    """
    if not os.getenv("GEMINI_API_KEY"):
        return "错误：未检测到 GEMINI_API_KEY，请先在 .env 文件中配置 GEMINI_API_KEY。"

    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    max_retries = 5
    retryable_errors = [
        "503",
        "UNAVAILABLE",
        "Server disconnected",
        "without sending a response",
        "timeout",
        "timed out",
        "connection",
        "Connection",
        "temporarily unavailable",
    ]

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=full_prompt,
            )

            return response.text or "模型没有返回文本结果。"

        except Exception as e:
            error_msg = str(e)

            if any(err in error_msg for err in retryable_errors):
                wait_time = 2 * (attempt + 1)
                time.sleep(wait_time)
                continue

            return format_llm_error(e)

    return "调用模型失败：当前模型请求量较高，已自动重试多次。请稍后再次点击分析。"


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


def run_agent_workflow(job_description: str, resume_text: str) -> dict:
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

    full_report = call_llm(prompt)

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
    }


def _build_error_result(message: str) -> dict:
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
    }


def run_rag_workflow(job_description: str, resume_text: str, source_name: str = "简历文本", top_k: int = 3) -> dict:
    """
    RAG 增强版分析流程：
    1. 根据岗位描述召回最相关的简历片段
    2. 将岗位描述和片段填入 RAG Prompt
    3. 调用 Gemini 生成完整报告
    4. 复用 extract_section 拆分为四个页面 Tab
    """
    try:
        retrieval_result = retrieve_relevant_chunks_with_sources(
            job_description=job_description,
            resume_text=resume_text,
            source_name=source_name,
            top_k=top_k,
        )
    except Exception as e:
        return _build_error_result(str(e))

    retrieved_context = retrieval_result.get("context", "")
    sources = retrieval_result.get("sources", [])

    if not retrieved_context.strip():
        return _build_error_result("没有召回到与岗位描述相关的简历片段，请检查简历文本是否足够完整。")

    prompt = RAG_ANALYSIS_PROMPT.format(
        job_description=job_description[:4000],
        retrieved_context=retrieved_context,
    )

    full_report = call_llm(prompt)

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
        "retrieved_context": retrieved_context,
        "retrieved_chunk_count": len(sources),
        "rag_sources": sources,
        "embedding_provider": retrieval_result.get("embedding_provider"),
        "rag_top_k": top_k,
        "rag_total_chunks": retrieval_result.get("total_chunks"),
    }
