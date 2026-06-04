import os
import time
from dotenv import load_dotenv
from google import genai

from prompts import (
    SYSTEM_PROMPT,
    COMPREHENSIVE_ANALYSIS_PROMPT,
    RAG_ANALYSIS_PROMPT,
)
from rag import retrieve_relevant_chunks

# 读取 .env 文件中的环境变量
load_dotenv()

# 初始化 Gemini 客户端
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 从 .env 读取模型名称，如果没有配置，则使用默认模型
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def call_llm(prompt: str) -> str:
    """
    调用 Gemini 大语言模型并返回文本结果。
    增加简单重试机制，用于处理 503 高负载等临时错误。
    """
    if not os.getenv("GEMINI_API_KEY"):
        return "错误：未检测到 GEMINI_API_KEY，请先在 .env 文件中配置 Gemini API Key。"

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

            return f"调用模型时出现错误：{e}"

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
    }


def run_rag_workflow(job_description: str, resume_text: str) -> dict:
    """
    RAG 增强版分析流程：
    1. 根据岗位描述召回最相关的简历片段
    2. 将岗位描述和片段填入 RAG Prompt
    3. 调用 Gemini 生成完整报告
    4. 复用 extract_section 拆分为四个页面 Tab
    """
    try:
        retrieved_context = retrieve_relevant_chunks(
            job_description=job_description,
            resume_text=resume_text,
            top_k=3,
        )
    except Exception as e:
        return _build_error_result(str(e))

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
        "retrieved_chunk_count": retrieved_context.count("[相关片段"),
    }
