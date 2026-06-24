# FastAPI 接口说明

启动服务：

```powershell
.\.venv\Scripts\python.exe -m uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
```

Swagger UI：`http://127.0.0.1:8000/docs`

## Streamlit API Client

`api_client.py` 使用 `requests` 封装 FastAPI 调用，并统一返回：

```json
{
  "success": true,
  "data": {},
  "error": null
}
```

它负责处理连接失败、超时、非 JSON、非 2xx 和后端 `success=false`。Streamlit 在 FastAPI 接口模式下使用：

- `check_api_health()` -> `GET /api/health`
- `call_rag_retrieve_api()` -> `POST /api/rag/retrieve`
- `call_agent_workflow_api()` -> `POST /api/agent/workflow`
- `call_markdown_report_api()` -> `POST /api/report/markdown`（当前页面仍保留本地 Markdown 导出）

当前 RAG API mode 只通过后端取得 retrieved chunks，后续 LLM 报告仍由 Streamlit 本地调用现有工具生成。Agent Workflow API mode 则使用后端返回的 analysis、steps 与 trace。

## GET `/api/health`

**用途**：确认 API 进程可访问。

**请求**：无请求体。

**返回示例**：

```json
{
  "status": "ok",
  "project": "AI Resume Agent",
  "version": "v1.6-fastapi-backend-mvp"
}
```

**真实 LLM 依赖**：无。

**当前限制**：只反映进程可用，不检查 Gemini、ChromaDB 或磁盘状态。

## POST `/api/rag/retrieve`

**用途**：从简历中召回与岗位 JD 相关的片段，可选轻量 Rerank。

**请求示例**：

```json
{
  "resume_text": "专业技能：Python、RAG、ChromaDB",
  "job_description": "招聘 Python RAG 开发实习生",
  "top_k": 3,
  "use_rerank": true
}
```

**返回示例**：

```json
{
  "success": true,
  "retrieved_chunks": [
    {
      "section": "skills",
      "chunk_index": 1,
      "distance": 0.42,
      "rerank_score": 7.2,
      "keyword_hits": ["Python", "RAG"]
    }
  ],
  "used_rerank": true,
  "rerank_method": "rule_based",
  "error": null
}
```

**真实 LLM 依赖**：无。Gemini embedding 模式可能依赖 Gemini API；local 模式不依赖。

**当前限制**：使用本地 ChromaDB collection，不提供多用户数据隔离和历史索引管理。

## POST `/api/agent/workflow`

**用途**：运行固定 Agent Workflow，返回分析、召回片段、Workflow Steps 和 Trace。

**请求示例**：

```json
{
  "resume_text": "项目经历：使用 Python 实现 RAG Demo",
  "job_description": "AI 应用开发岗位",
  "top_k": 3,
  "use_rag": true,
  "use_rerank": true,
  "use_mock_llm": false
}
```

**返回示例**：

```json
{
  "success": true,
  "analysis": "## 岗位要求分析\n...",
  "retrieved_chunks": [],
  "workflow_steps": [],
  "trace": {"run_id": "20260624T...", "final_status": "success"},
  "llm_mode": "gemini",
  "error": null
}
```

**真实 LLM 依赖**：默认依赖 Gemini。设置 `use_mock_llm=true` 时使用确定性 mock，仅用于 smoke test 和接口链路验证。

**当前限制**：固定工具流程、同步执行；没有复杂 planning、任务队列、历史记录或权限系统。

## POST `/api/report/markdown`

**用途**：将分析文本和 metadata 组装成 Markdown。

**请求示例**：

```json
{
  "analysis": "## 岗位要求分析\n需要 Python 与 RAG。",
  "metadata": {"mode": "agent_workflow"}
}
```

**返回示例**：

```json
{
  "success": true,
  "markdown": "# AI Resume Match Report\n...",
  "error": null
}
```

**真实 LLM 依赖**：无。

**当前限制**：返回 Markdown 内容，不提供文件存储、版本管理和访问权限。
