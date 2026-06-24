import os
import time
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompts import SYSTEM_PROMPT


load_dotenv()

SUPPORTED_LLM_PROVIDERS = {"gemini", "deepseek", "openai_compatible", "mock"}


@dataclass
class LLMResult:
    success: bool
    provider: str
    model: str | None
    text: str
    error: str | None = None
    raw: dict[str, Any] | None = None
    fallback_used: bool = False
    original_provider: str | None = None


@dataclass
class ProviderHealthResult:
    provider: str
    model: str | None
    available: bool
    message: str
    latency_ms: float | None = None
    error: str | None = None


def normalize_provider_error(error: Exception | str) -> str:
    """Convert common provider failures into short user-facing summaries."""
    message = str(error or "未知模型错误。")
    lowered = message.lower()
    if "api_key" in lowered or "api key" in lowered or "unauthenticated" in lowered:
        return "模型 API Key 缺失、无效或没有权限。"
    if "base_url" in lowered or "base url" in lowered:
        return "模型服务 Base URL 未配置。"
    if "timeout" in lowered or "超时" in message or "timed out" in lowered:
        return "模型服务请求超时。"
    if "connection" in lowered or "无法连接" in message or "连接" in message:
        return "无法连接模型服务，请检查网络和 Base URL。"
    if "http" in lowered and any(char.isdigit() for char in message):
        return f"模型服务返回 HTTP 错误：{message}"
    if "choices[0]" in lowered or "非 json" in lowered or "响应" in message:
        return f"模型服务响应结构异常：{message}"
    if "不支持" in message or "unsupported" in lowered:
        return f"不支持的模型 Provider：{message}"
    return message


def get_llm_provider_from_env() -> str:
    """Read and normalise the LLM_PROVIDER env var.  Does *not* check for API keys —
    missing-key handling belongs in generate_with_llm() so the fallback path can record
    original_provider / provider_error correctly."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in SUPPORTED_LLM_PROVIDERS:
        return "gemini"
    return provider


def _mock_result(
    model: str | None = None,
    original_provider: str | None = None,
    provider_error: str | None = None,
) -> LLMResult:
    mock_model = model or "mock-structured-v1"
    text = """## 岗位要求分析
岗位需要结合 JD 判断核心职责、技术要求和项目经验要求。

## 个人能力分析
### 能力匹配
候选人提供的信息体现了部分岗位相关技能和项目实践。

### 差距分析
未在输入中明确体现的能力应标记为待补充，并在面试前准备具体案例。

## 匹配度分析
当前信息可以完成基础匹配分析，但结论仅用于流程测试，不代表真实模型质量。

## 简历优化建议
补充项目背景、个人职责、技术方案、量化结果和问题排查过程。
"""
    return LLMResult(
        success=True,
        provider="mock",
        model=mock_model,
        text=text,
        error=provider_error,
        raw={
            "mode": "deterministic_mock",
            "result_error": None,
            "provider_error": provider_error,
            "original_provider": original_provider,
            "fallback_provider": "mock" if original_provider else None,
        },
        fallback_used=original_provider is not None,
        original_provider=original_provider,
    )


def _format_gemini_error(error: Exception) -> str:
    message = str(error)
    lowered = message.lower()
    if "429" in message or "resource_exhausted" in lowered or "quota" in lowered:
        return "Gemini API 额度不足或请求过多，请稍后重试或检查配额。"
    if "location is not supported" in lowered:
        return "当前网络出口地区可能不支持 Gemini API。"
    if "api key" in lowered or "permission_denied" in lowered or "unauthenticated" in lowered:
        return "Gemini API Key 无效或缺少权限。"
    if "model" in lowered and ("not found" in lowered or "invalid" in lowered):
        return "Gemini 模型名称不可用，请检查 GEMINI_MODEL。"
    return f"Gemini 调用失败：{message}"


def _generate_gemini(prompt: str, model: str | None, timeout: int, max_retries: int = 5) -> LLMResult:
    api_key = os.getenv("GEMINI_API_KEY")
    selected_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if not api_key:
        return LLMResult(False, "gemini", selected_model, "", "未配置 GEMINI_API_KEY。")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=max(timeout, 1) * 1000),
    )
    retryable = ("503", "unavailable", "timeout", "connection", "temporarily unavailable")
    last_error: Exception | None = None
    total_attempts = max(max_retries, 1)
    for attempt in range(total_attempts):
        try:
            response = client.models.generate_content(model=selected_model, contents=prompt)
            text = response.text or ""
            if not text:
                return LLMResult(False, "gemini", selected_model, "", "Gemini 没有返回文本结果。")
            return LLMResult(True, "gemini", selected_model, text)
        except Exception as exc:
            last_error = exc
            if any(keyword in str(exc).lower() for keyword in retryable) and attempt < total_attempts - 1:
                time.sleep(2 * (attempt + 1))
                continue
            break
    return LLMResult(False, "gemini", selected_model, "", _format_gemini_error(last_error or RuntimeError("unknown error")))


def _call_chat_completions(
    prompt: str,
    model: str,
    timeout: int,
    api_key: str,
    base_url: str,
    provider_name: str,
) -> LLMResult:
    """Shared Chat Completions HTTP logic — used by openai_compatible and deepseek."""
    endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=timeout,
        )
    except requests.Timeout:
        return LLMResult(False, provider_name, model, "", f"模型请求超时（{timeout} 秒）。")
    except requests.ConnectionError:
        return LLMResult(False, provider_name, model, "", f"无法连接 {provider_name} 服务。")
    except requests.RequestException as exc:
        return LLMResult(False, provider_name, model, "", f"模型请求失败：{exc}")

    try:
        raw = response.json()
    except ValueError:
        return LLMResult(
            False,
            provider_name,
            model,
            "",
            f"模型服务返回非 JSON 响应（HTTP {response.status_code}）。",
        )
    if not response.ok:
        api_error = raw.get("error") if isinstance(raw, dict) else None
        if isinstance(api_error, dict):
            api_error = api_error.get("message") or str(api_error)
        return LLMResult(
            False,
            provider_name,
            model,
            "",
            f"模型服务返回 HTTP {response.status_code}：{api_error or 'unknown error'}",
            raw=raw if isinstance(raw, dict) else None,
        )

    try:
        text = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return LLMResult(False, provider_name, model, "", "响应缺少 choices[0].message.content。", raw=raw)
    return LLMResult(True, provider_name, model, str(text), raw=raw)


def _generate_openai_compatible(
    prompt: str,
    model: str | None,
    timeout: int,
) -> LLMResult:
    api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "").strip().rstrip("/")
    selected_model = model or os.getenv("OPENAI_COMPATIBLE_MODEL", "deepseek-chat")
    if not api_key:
        return LLMResult(
            False,
            "openai_compatible",
            selected_model,
            "",
            "未配置 OPENAI_COMPATIBLE_API_KEY。",
        )
    if not base_url:
        return LLMResult(
            False,
            "openai_compatible",
            selected_model,
            "",
            "未配置 OPENAI_COMPATIBLE_BASE_URL。",
        )
    return _call_chat_completions(prompt, selected_model, timeout, api_key, base_url, "openai_compatible")


def _generate_deepseek(
    prompt: str,
    model: str | None,
    timeout: int,
) -> LLMResult:
    """DeepSeek native provider — reuses Chat Completions protocol under its own name."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
    selected_model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    if not api_key:
        return LLMResult(
            False,
            "deepseek",
            selected_model,
            "",
            "未配置 DEEPSEEK_API_KEY。",
        )
    return _call_chat_completions(prompt, selected_model, timeout, api_key, base_url, "deepseek")


def generate_with_llm(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    use_mock: bool = False,
    timeout: int = 60,
    fallback_to_mock: bool = True,
    max_retries: int = 5,
) -> LLMResult:
    """Generate text through Gemini, an OpenAI-compatible endpoint, or deterministic mock.

    max_retries only applies to the Gemini provider (per-request retry on transient
    errors).  Health-check callers should pass a small value (0 or 1) to avoid hanging.
    """
    selected_provider = "mock" if use_mock else (provider or get_llm_provider_from_env())
    selected_provider = selected_provider.strip().lower()
    if selected_provider not in SUPPORTED_LLM_PROVIDERS:
        return LLMResult(
            False,
            selected_provider,
            model,
            "",
            f"不支持的 LLM provider：{selected_provider}。",
        )

    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    if selected_provider == "mock":
        return _mock_result(model)
    if selected_provider == "gemini":
        result = _generate_gemini(full_prompt, model, timeout, max_retries=max_retries)
    elif selected_provider == "deepseek":
        result = _generate_deepseek(full_prompt, model, timeout)
    else:
        result = _generate_openai_compatible(full_prompt, model, timeout)

    if result.success or not fallback_to_mock:
        if result.error:
            result.error = normalize_provider_error(result.error)
        return result

    error_summary = normalize_provider_error(result.error or "模型调用失败。")
    return _mock_result(
        original_provider=selected_provider,
        provider_error=error_summary,
    )


def check_llm_provider_health(
    provider: str | None = None,
    model: str | None = None,
    timeout: int = 10,
    use_mock: bool = False,
) -> ProviderHealthResult:
    """Check configuration first, then make a short request only when configured."""
    selected_provider = "mock" if use_mock else (provider or get_llm_provider_from_env())
    selected_provider = selected_provider.strip().lower()
    started = perf_counter()

    if selected_provider == "mock":
        return ProviderHealthResult(
            provider="mock",
            model=model or "mock-structured-v1",
            available=True,
            message="Mock Provider 可用，不依赖外部服务。",
            latency_ms=round((perf_counter() - started) * 1000, 2),
        )
    if selected_provider not in SUPPORTED_LLM_PROVIDERS:
        error = normalize_provider_error(f"不支持的 LLM provider：{selected_provider}")
        return ProviderHealthResult(selected_provider, model, False, "Provider 不可用。", 0.0, error)
    if selected_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        error = normalize_provider_error("未配置 GEMINI_API_KEY。")
        return ProviderHealthResult("gemini", model or os.getenv("GEMINI_MODEL"), False, "Gemini 配置不完整。", 0.0, error)
    if selected_provider == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        error = normalize_provider_error("未配置 DEEPSEEK_API_KEY。")
        return ProviderHealthResult("deepseek", model or os.getenv("DEEPSEEK_MODEL"), False, "DeepSeek 配置不完整。", 0.0, error)
    if selected_provider == "openai_compatible":
        if not os.getenv("OPENAI_COMPATIBLE_API_KEY"):
            error = normalize_provider_error("未配置 OPENAI_COMPATIBLE_API_KEY。")
            return ProviderHealthResult(selected_provider, model or os.getenv("OPENAI_COMPATIBLE_MODEL"), False, "OpenAI-compatible 配置不完整。", 0.0, error)
        if not os.getenv("OPENAI_COMPATIBLE_BASE_URL"):
            error = normalize_provider_error("未配置 OPENAI_COMPATIBLE_BASE_URL。")
            return ProviderHealthResult(selected_provider, model or os.getenv("OPENAI_COMPATIBLE_MODEL"), False, "OpenAI-compatible 配置不完整。", 0.0, error)

    result = generate_with_llm(
        "请只回复 ok",
        provider=selected_provider,
        model=model,
        timeout=timeout,
        fallback_to_mock=False,
        max_retries=1,
    )
    latency_ms = round((perf_counter() - started) * 1000, 2)
    return ProviderHealthResult(
        provider=selected_provider,
        model=result.model,
        available=result.success,
        message="Provider 健康检查通过。" if result.success else "Provider 健康检查失败。",
        latency_ms=latency_ms,
        error=normalize_provider_error(result.error) if result.error else None,
    )
