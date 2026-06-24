import os
import sys

import requests


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = 60


def _check_response(name: str, response: requests.Response) -> dict:
    response.raise_for_status()
    data = response.json()
    if data.get("success") is False:
        raise RuntimeError(data.get("error") or f"{name} returned success=false")
    print(f"{name}: PASS")
    return data


def main() -> int:
    resume_text = "专业技能：Python、RAG、ChromaDB。项目经历：实现 Agent 简历分析 Demo。"
    job_description = "招聘 Python RAG Agent 应用开发实习生。"

    try:
        health = requests.get(f"{BASE_URL}/api/health", timeout=TIMEOUT_SECONDS)
        health.raise_for_status()
        if health.json().get("status") != "ok":
            raise RuntimeError("health endpoint did not return status=ok")
        print("GET /api/health: PASS")

        docs = requests.get(f"{BASE_URL}/docs", timeout=TIMEOUT_SECONDS)
        docs.raise_for_status()
        if "swagger" not in docs.text.lower():
            raise RuntimeError("/docs did not return the Swagger UI page")
        print("GET /docs: PASS")

        _check_response(
            "POST /api/rag/retrieve",
            requests.post(
                f"{BASE_URL}/api/rag/retrieve",
                json={
                    "resume_text": resume_text,
                    "job_description": job_description,
                    "top_k": 2,
                    "use_rerank": True,
                },
                timeout=TIMEOUT_SECONDS,
            ),
        )

        workflow = _check_response(
            "POST /api/agent/workflow",
            requests.post(
                f"{BASE_URL}/api/agent/workflow",
                json={
                    "resume_text": resume_text,
                    "job_description": job_description,
                    "top_k": 2,
                    "use_rag": True,
                    "use_rerank": True,
                    "llm_provider": "mock",
                    "use_mock_llm": True,
                },
                timeout=TIMEOUT_SECONDS,
            ),
        )
        if not workflow.get("trace", {}).get("run_id"):
            raise RuntimeError("Agent workflow response is missing trace.run_id")
        if workflow.get("llm_provider") != "mock":
            raise RuntimeError("Agent workflow did not use the requested mock provider")

        _check_response(
            "POST /api/report/markdown",
            requests.post(
                f"{BASE_URL}/api/report/markdown",
                json={
                    "analysis": workflow.get("analysis", "Smoke test analysis"),
                    "metadata": {"mode": "agent_workflow", "llm_mode": "mock"},
                },
                timeout=TIMEOUT_SECONDS,
            ),
        )
    except requests.ConnectionError:
        print(
            f"API service is not running at {BASE_URL}. Start run_api.bat or uvicorn first.",
            file=sys.stderr,
        )
        return 1
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        print(f"API smoke test failed: {exc}", file=sys.stderr)
        return 1

    print("api_smoke_test.py: all endpoints passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
