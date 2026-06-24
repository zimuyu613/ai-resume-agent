from typing import Any

import requests


DEFAULT_TIMEOUT_SECONDS = 30
AGENT_TIMEOUT_SECONDS = 120


def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def _call_api(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    base_url = _normalize_base_url(base_url)
    if not base_url:
        return {"success": False, "data": None, "error": "API Base URL 不能为空。"}

    try:
        response = requests.request(
            method=method,
            url=f"{base_url}{path}",
            json=payload,
            timeout=timeout,
        )
    except requests.ConnectionError:
        return {
            "success": False,
            "data": None,
            "error": "FastAPI 服务未启动，请先运行 run_api.bat 或 uvicorn 命令。",
        }
    except requests.Timeout:
        return {
            "success": False,
            "data": None,
            "error": f"FastAPI 请求超时（{timeout} 秒），请检查后端或模型服务状态。",
        }
    except requests.RequestException as exc:
        return {"success": False, "data": None, "error": f"FastAPI 请求失败：{exc}"}

    try:
        data = response.json()
    except ValueError:
        return {
            "success": False,
            "data": None,
            "error": f"FastAPI 返回了非 JSON 响应（HTTP {response.status_code}）。",
        }

    if not response.ok:
        error = data.get("error") or data.get("detail") or f"HTTP {response.status_code}"
        return {"success": False, "data": data, "error": str(error)}

    if isinstance(data, dict) and data.get("success") is False:
        return {
            "success": False,
            "data": data,
            "error": str(data.get("error") or "FastAPI 返回 success=false。"),
        }

    return {"success": True, "data": data, "error": None}


def check_api_health(base_url: str) -> dict[str, Any]:
    return _call_api("GET", base_url, "/api/health", timeout=10)


def call_rag_retrieve_api(
    base_url: str,
    resume_text: str,
    job_description: str,
    top_k: int = 5,
    use_rerank: bool = False,
) -> dict[str, Any]:
    return _call_api(
        "POST",
        base_url,
        "/api/rag/retrieve",
        payload={
            "resume_text": resume_text,
            "job_description": job_description,
            "top_k": top_k,
            "use_rerank": use_rerank,
        },
    )


def call_agent_workflow_api(
    base_url: str,
    resume_text: str,
    job_description: str,
    top_k: int = 5,
    use_rag: bool = True,
    use_rerank: bool = False,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    use_mock_llm: bool = False,
) -> dict[str, Any]:
    return _call_api(
        "POST",
        base_url,
        "/api/agent/workflow",
        payload={
            "resume_text": resume_text,
            "job_description": job_description,
            "top_k": top_k,
            "use_rag": use_rag,
            "use_rerank": use_rerank,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "use_mock_llm": use_mock_llm,
        },
        timeout=AGENT_TIMEOUT_SECONDS,
    )


def call_markdown_report_api(
    base_url: str,
    analysis: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _call_api(
        "POST",
        base_url,
        "/api/report/markdown",
        payload={"analysis": analysis, "metadata": metadata or {}},
    )
