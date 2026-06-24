# AI Resume Agent 最终检查清单

本清单用于发布前、本地演示前和面试前快速自检。项目定位是 AI 应用工程化 Demo，以及 RAG Workflow / Agent Workflow MVP。

## 本地运行检查

- [ ] `run_app.bat` 或 Streamlit 命令可以启动页面。
- [ ] `run_api.bat` 或 Uvicorn 命令可以启动 FastAPI。
- [ ] `http://127.0.0.1:8000/docs` 可以打开。
- [ ] `.\.venv\Scripts\python.exe simple_test.py` 通过。
- [ ] `.\.venv\Scripts\python.exe eval_runner.py` 通过。
- [ ] Eval Runner 生成 `eval_result_<timestamp>.json`。
- [ ] Eval Runner 生成 `eval_summary_<timestamp>.md`。
- [ ] 控制台和结果中可以看到 Recall@K / MRR。
- [ ] 可以看到普通 RAG 与 RAG + Rerank 的 improved / same / worse 对比。
- [ ] API 启动后，`.\.venv\Scripts\python.exe api_smoke_test.py` 通过。
- [ ] 下列语法检查通过：

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py agent.py agent_workflow.py tools.py trace_utils.py rerank_utils.py rag_eval_utils.py simple_test.py eval_runner.py api_server.py api_smoke_test.py
```

## Demo 展示检查

- [ ] 普通 LLM 分析可以生成四部分结果。
- [ ] RAG 检索增强分析可以展示召回片段和 metadata。
- [ ] Agent Workflow 分析可以展示 Workflow Steps。
- [ ] 轻量 Rerank 开启后可以展示 `rerank_score`、`keyword_hits` 和 `section_bonus`。
- [ ] Trace 摘要、步骤、JSON 展示和下载正常。
- [ ] Markdown 报告可以下载。
- [ ] FastAPI `/docs` 中四个接口可以查看和试用。

## 安全检查

- [ ] `.env` 未提交。
- [ ] 代码、文档和截图中没有真实 API Key。
- [ ] `eval_results/` 未提交。
- [ ] `outputs/traces/` 未提交。
- [ ] `chroma_db/` 和其他本地向量缓存未提交。
- [ ] 示例简历、报告和 Eval case 不包含真实个人隐私。

## 面试讲解检查

- [ ] 能讲清楚项目目标与用户使用流程。
- [ ] 能讲清楚 RAG 的 chunk、embedding、检索和 Prompt 流程。
- [ ] 能讲清楚 Tool Calling 工具层的职责。
- [ ] 能讲清楚固定 Agent Workflow 与复杂自主 Agent 的区别。
- [ ] 能讲清楚 Trace 如何帮助定位问题。
- [ ] 能讲清楚 Eval Runner 验证了什么、没有验证什么。
- [ ] 能讲清楚 rule-based Rerank 的评分逻辑与限制。
- [ ] 能讲清楚 FastAPI 后端如何复用现有业务模块。
- [ ] 能主动说明项目边界和合理的后续方向。
