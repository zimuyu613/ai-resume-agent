# AI Resume Agent 面试讲解笔记

## 1 分钟项目介绍

这是一个 AI 简历与岗位匹配分析助手，定位是 AI 应用工程化 Demo。用户上传 txt、pdf 或 docx 简历并输入岗位 JD 后，可以选择普通 LLM、RAG 或 Agent Workflow 三种分析模式。RAG 会按简历 section 切分文本，生成 embedding 后写入 ChromaDB，再召回与 JD 相关的片段。项目还加入了轻量 Tool Calling、Trace、rule-based Rerank、Eval Runner 和 FastAPI 接口层。它适合展示一条可运行、可解释、可基础验证的 AI 应用链路，但不是生产级 Agent 平台。

## 3 分钟项目介绍

项目解决的问题是：直接把整份简历和 JD 交给大模型，分析过程不容易解释，也难以判断模型参考了哪些经历。因此我先实现普通 LLM 模式作为基线，再实现 RAG 模式。

RAG 不是简单按固定长度切全文，而是先识别技能、项目、实习、教育等 section，再在 section 内切 chunk。每个 chunk 会携带 section、位置、来源和长度等 metadata，随后生成 embedding 并存入 ChromaDB。岗位 JD 作为 query 召回 top_k 片段，页面会展示 distance 和片段内容。

在此基础上，我把解析、检索、LLM 分析和 Markdown 导出封装成工具，并用固定 Agent Workflow 串联 RAG 工具和分析工具。为了方便排查，我记录每次运行的 run_id、输入摘要、工具耗时、输出摘要和错误。轻量 Rerank 会结合 JD 关键词、section bonus 和原始 distance 做二次排序。

项目还提供不依赖真实 Gemini 的 Eval Runner，用脱敏样例验证召回、Workflow、Trace 和输出结构；FastAPI 则把现有能力暴露为 HTTP 接口。当前仍是 MVP，没有复杂 planning、多 Agent、长期记忆、权限、历史数据库或生产部署。

## 技术架构讲解

- 展示层：`app.py` 使用 Streamlit，实现输入、模式选择和结果展示。
- API 层：`api_server.py` 使用 FastAPI，复用工具与 Workflow。
- 模型层：`agent.py` 负责 Gemini 调用、Prompt 和报告拆分。
- 检索层：`rag.py` 负责 section-aware chunking、embedding 和 ChromaDB。
- Agent 层：`tools.py` 封装工具，`agent_workflow.py` 编排固定流程。
- 可解释层：`trace_utils.py` 记录 Trace，`rerank_utils.py` 提供可解释排序。
- 验证层：`simple_test.py` 做快速检查，`eval_runner.py` 做工程型 case 评测。

## RAG 流程讲解

1. 解析简历文本。
2. 识别 `skills`、`project_experience` 等 section。
3. 在 section 内切 chunk 并保留 metadata。
4. 使用配置的 embedding provider 生成向量。
5. 写入 ChromaDB。
6. 将岗位 JD 转为 query embedding，召回 top_k 候选片段。
7. 可选执行轻量 Rerank。
8. 把最终片段加入 RAG Prompt，调用 LLM 生成分析。

section-aware chunking 的目的，是减少一个 chunk 同时混入多个简历模块，使召回来源更容易解释。当前 section 识别基于规则，不是复杂 NLP。

## Tool Calling / Agent Workflow 讲解

`tools.py` 统一返回 `ToolResult`，包含成功状态、工具名、消息、数据和错误。当前工具覆盖简历解析、RAG 检索、LLM 分析和 Markdown 导出。

`agent_workflow.py` 按固定顺序调用 RAG 检索工具和 LLM 分析工具。它体现了工具封装与调用链路，但没有动态 planning 和自主工具选择，所以更准确的说法是 Agent Workflow MVP。

## Trace / Observability 讲解

每次 Agent Workflow 会生成 run_id，并记录整体耗时、输入长度、top_k、embedding provider、fallback、rerank 状态和最终结果。每一步还记录输入输出摘要、开始结束时间、耗时和错误。

Trace 的价值是帮助判断问题发生在召回、排序还是 LLM 阶段。当前实现是本地 JSON 和页面展示，不等同于 OpenTelemetry 或 LangSmith。

## Eval Runner 讲解

`eval_cases/` 中每个 case 包含简历、JD 和 expected.json。Eval Runner 检查 RAG 是否召回片段、是否命中预期 section 和关键词、Agent Workflow 与 Trace 是否完整，以及非 RAG 路径是否正常。

为了可复现，Eval 默认使用 local embedding 和 mock LLM。因此它验证的是工程链路稳定性，不代表真实 Gemini 回答质量。

## Rerank 讲解

ChromaDB 首次召回后，rule-based Rerank 根据三类信号排序：JD 关键词重合数量、技能/项目/实习 section bonus、原始 distance 转换分数。页面和 Trace 会展示评分依据。

它依赖少、容易解释，但不是 cross-encoder，也不能替代训练过的语义 reranker。

## FastAPI 后端讲解

FastAPI 提供 health、RAG retrieve、Agent Workflow 和 Markdown report 四个接口。接口层只做请求校验和结果映射，核心逻辑继续复用 `tools.py` 与 `agent_workflow.py`。

这形成了前后端分离接口雏形，但 Streamlit 当前仍直接调用 Python 模块，不是完整前后端解耦。后端也没有登录、数据库、权限、队列和生产部署。

## 项目边界说明

- AI 应用工程化 Demo，不是生产级系统。
- RAG Workflow / Agent Workflow MVP，不是自主多 Agent 平台。
- 没有复杂 planning、多 Agent、长期记忆和用户权限。
- FastAPI 是最小接口层，没有历史数据库和异步任务。
- Trace 是轻量本地实现。
- Eval 是工程型基础评测，不代表严格模型质量。
- local hash embedding 用于稳定 fallback，不代表最佳检索效果。

## 后续优化方向

1. 使用标注数据增加 Recall@K、MRR 等检索指标。
2. 在数据规模提升后评估 cross-encoder reranker。
3. 将 Streamlit 改为调用 FastAPI，进一步解耦展示层与业务层。
4. 增加多模型 provider、超时、限流和统一错误码。
5. 在真实需求出现后再评估 LangGraph、复杂规划和状态持久化。
