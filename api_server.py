from typing import Any

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from agent_workflow import run_resume_agent_workflow
from tools import export_markdown_tool, rag_retrieve_tool


PROJECT_VERSION = "v1.6-fastapi-backend-mvp"

app = FastAPI(
    title="AI Resume Agent API",
    version=PROJECT_VERSION,
    description="Minimal API wrapper around the existing RAG and Agent Workflow modules.",
)


class RagRetrieveRequest(BaseModel):
    resume_text: str
    job_description: str
    top_k: int = Field(default=5, ge=1, le=20)
    use_rerank: bool = False

    @field_validator("resume_text", "job_description")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class AgentWorkflowRequest(BaseModel):
    resume_text: str
    job_description: str
    top_k: int = Field(default=5, ge=1, le=20)
    use_rag: bool = True
    use_rerank: bool = False
    use_mock_llm: bool = False

    @field_validator("resume_text", "job_description")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class MarkdownReportRequest(BaseModel):
    analysis: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("analysis")
    @classmethod
    def validate_analysis(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, exc: RequestValidationError) -> JSONResponse:
    messages = [
        f"{'.'.join(str(part) for part in error['loc'][1:])}: {error['msg']}"
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"success": False, "error": "Invalid request: " + "; ".join(messages)},
    )


def _mock_llm_for_smoke_test(_prompt: str) -> str:
    return """## 岗位要求分析
岗位要求 Python、RAG 和 Agent 应用开发能力。

## 个人能力分析
候选人的示例经历体现了基础项目能力。

## 匹配度分析
当前信息与岗位存在基础匹配。

## 简历优化建议
补充项目指标、问题排查过程和实际效果。
"""


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "project": "AI Resume Agent",
        "version": PROJECT_VERSION,
    }


@app.post("/api/rag/retrieve")
def retrieve_rag(request: RagRetrieveRequest) -> dict[str, Any]:
    try:
        result = rag_retrieve_tool(
            resume_text=request.resume_text,
            job_description=request.job_description,
            top_k=request.top_k,
            use_rerank=request.use_rerank,
        )
        return {
            "success": result.success,
            "retrieved_chunks": result.data.get("chunks", []),
            "used_rerank": result.data.get("used_rerank", False),
            "rerank_method": result.data.get("rerank_method"),
            "error": result.error,
        }
    except Exception as exc:
        return {
            "success": False,
            "retrieved_chunks": [],
            "used_rerank": request.use_rerank,
            "rerank_method": "rule_based" if request.use_rerank else None,
            "error": f"RAG retrieval failed: {exc}",
        }


@app.post("/api/agent/workflow")
def run_agent(request: AgentWorkflowRequest) -> dict[str, Any]:
    try:
        result = run_resume_agent_workflow(
            resume_text=request.resume_text,
            job_description=request.job_description,
            top_k=request.top_k,
            use_rag=request.use_rag,
            use_rerank=request.use_rerank,
            llm_callable=_mock_llm_for_smoke_test if request.use_mock_llm else None,
        )
        return {
            "success": result.get("success", False),
            "analysis": result.get("analysis", ""),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "workflow_steps": result.get("workflow_steps", []),
            "trace": result.get("trace", {}),
            "llm_mode": "mock" if request.use_mock_llm else "gemini",
            "error": result.get("error"),
        }
    except Exception as exc:
        return {
            "success": False,
            "analysis": "",
            "retrieved_chunks": [],
            "workflow_steps": [],
            "trace": {},
            "llm_mode": "mock" if request.use_mock_llm else "gemini",
            "error": f"Agent Workflow failed: {exc}",
        }


@app.post("/api/report/markdown")
def create_markdown_report(request: MarkdownReportRequest) -> dict[str, Any]:
    try:
        result = export_markdown_tool(
            analysis_result=request.analysis,
            metadata=request.metadata,
        )
        return {
            "success": result.success,
            "markdown": result.data.get("markdown", ""),
            "error": result.error,
        }
    except Exception as exc:
        return {
            "success": False,
            "markdown": "",
            "error": f"Markdown report generation failed: {exc}",
        }
