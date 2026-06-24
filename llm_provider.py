import os
import time
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv
from google import genai

from prompts import SYSTEM_PROMPT


load_dotenv()

SUPPORTED_LLM_PROVIDERS = {"gemini", "openai_compatible", "mock"}


@dataclass
class LLMResult:
    success: bool
    provider: str
    model: str | None
    text: str
    error: str | None = None
    raw: dict[str, Any] | None = None


def get_llm_provider_from_env() -> str:
    """Resolve the default provider; use mock when the configured provider has no key."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in SUPPORTED_LLM_PROVIDERS:
        provider = "gemini"
    if provider == "mock":
        return "mock"
    if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        return "mock"
    if provider == "openai_compatible" and not os.getenv("OPENAI_COMPATIBLE_API_KEY"):
        return "mock"
    return provider


def _mock_result(model: str | None = None) -> LLMResult:
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
        raw={"mode": "deterministic_mock"},
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


def _generate_gemini(prompt: str, model: str | None) -> LLMResult:
    api_key = os.getenv("GEMINI_API_KEY")
    selected_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if not api_key:
        return LLMResult(False, "gemini", selected_model, "", "未配置 GEMINI_API_KEY。")

    client = genai.Client(api_key=api_key)
    retryable = ("503", "unavailable", "timeout", "connection", "temporarily unavailable")
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = client.models.generate_content(model=selected_model, contents=prompt)
            text = response.text or ""
            if not text:
                return LLMResult(False, "gemini", selected_model, "", "Gemini 没有返回文本结果。")
            return LLMResult(True, "gemini", selected_model, text)
        except Exception as exc:
            last_error = exc
            if any(keyword in str(exc).lower() for keyword in retryable) and attempt < 4:
                time.sleep(2 * (attempt + 1))
                continue
            break
    return LLMResult(False, "gemini", selected_model, "", _format_gemini_error(last_error or RuntimeError("unknown error")))


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

    endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": selected_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=timeout,
        )
    except requests.Timeout:
        return LLMResult(False, "openai_compatible", selected_model, "", f"模型请求超时（{timeout} 秒）。")
    except requests.ConnectionError:
        return LLMResult(False, "openai_compatible", selected_model, "", "无法连接 OpenAI-compatible 服务。")
    except requests.RequestException as exc:
        return LLMResult(False, "openai_compatible", selected_model, "", f"模型请求失败：{exc}")

    try:
        raw = response.json()
    except ValueError:
        return LLMResult(
            False,
            "openai_compatible",
            selected_model,
            "",
            f"模型服务返回非 JSON 响应（HTTP {response.status_code}）。",
        )
    if not response.ok:
        api_error = raw.get("error") if isinstance(raw, dict) else None
        if isinstance(api_error, dict):
            api_error = api_error.get("message") or str(api_error)
        return LLMResult(
            False,
            "openai_compatible",
            selected_model,
            "",
            f"模型服务返回 HTTP {response.status_code}：{api_error or 'unknown error'}",
            raw=raw if isinstance(raw, dict) else None,
        )

    try:
        text = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return LLMResult(False, "openai_compatible", selected_model, "", "响应缺少 choices[0].message.content。", raw=raw)
    return LLMResult(True, "openai_compatible", selected_model, str(text), raw=raw)


def generate_with_llm(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    use_mock: bool = False,
    timeout: int = 60,
) -> LLMResult:
    """Generate text through Gemini, an OpenAI-compatible endpoint, or deterministic mock."""
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
        return _generate_gemini(full_prompt, model)
    return _generate_openai_compatible(full_prompt, model, timeout)
