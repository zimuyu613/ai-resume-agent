# Lightweight Agent Harness

## 什么是 Harness

在本项目中，Harness 是指围绕 RAG 检索和 LLM 分析构建的一套**轻量工程增强层**。它的目标不是打造复杂的自主 Agent，而是：

- 提高分析流程的**可解释性**（Reviewer 告诉你报告哪部分缺失）。
- 提高检索的**稳定性**（检索质量不足时自动重试一次）。
- 将观测、审核和边界控制**显式化**，便于面试讲解和工程调试。

当前 Harness 是教学和 MVP 级别的实现，不等同于 LangGraph、AutoGPT 或生产级 Agent 框架。

## Harness 组成部分

| 层级 | 文件 / 模块 | 职责 |
|------|------------|------|
| Tool Layer | `tools.py` | `parse_resume_tool`, `rag_retrieve_tool`, `llm_match_analysis_tool`, `export_markdown_tool`, **`review_report_tool`** |
| Workflow Layer | `agent_workflow.py` | 固定 Tool Calling 编排、**query refinement loop**、Trace 生命周期 |
| Model Provider Layer | `llm_provider.py` | Gemini / DeepSeek / OpenAI Compatible / Mock 统一入口、Health Check、Fallback |
| Trace Layer | `trace_utils.py` | `TraceStep`, `WorkflowTrace` dataclass、JSON 序列化、本地保存 |
| Eval Layer | `eval_runner.py`, `rag_eval_utils.py` | Recall@K, MRR, gold evidence 评测、RAG vs Rerank 对比、Workflow 工程检查 |
| Fallback Layer | `llm_provider.py` | `fallback_to_mock` 降级、错误标准化（`normalize_provider_error`）、`LLMResult` 元数据 |

## Tool Layer

所有工具通过 `ToolResult(success, tool_name, message, data, error)` 统一返回。`agent_workflow.py` 的 `_timed_tool_step` 函数为每个工具调用生成 `TraceStep`，记录耗时、输入摘要和输出摘要。

### review_report_tool

**输入**：`job_description`、`retrieved_chunks`、`analysis_text`

**输出**：
- `review_passed: bool` — 是否通过审核
- `missing_points: list[str]` — 缺失的结构部分
- `evidence_usage: str` — 证据使用摘要
- `risk_notes: list[str]` — 风险提示
- `improvement_suggestions: list[str]` — 改进建议
- `review_summary: str` — 审核总结

**审核逻辑**（纯规则，不调用真实 LLM）：
1. 检查 `analysis_text` 是否为空
2. 检查是否包含四个关键结构：`岗位要求分析`、`能力分析`、`匹配`、`优化`
3. 检查 `retrieved_chunks` 是否存在
4. 检查分析文本中是否出现项目、技能、实习等证据关键词

## Workflow Layer

### 固定 Workflow 流程

```
RAG retrieve (rag_retrieve_tool)
  └→ [可选] Query Refinement（最多 1 次重试）
  └→ LLM match analysis (llm_match_analysis_tool)
      └→ Review (review_report_tool)
```

### Bounded Query Refinement Loop

函数 `evaluate_retrieval_quality(retrieved_chunks)` 判断第一次检索质量：

- 如果 `retrieved_chunks` 为空 → `quality = "low"`
- 如果 top chunks 没有 `keyword_hits` 且 `rerank_score` 低于 0.5 → `quality = "low"`
- 否则 → `quality = "ok"`

当 `quality = "low"` 且 `use_rag = True` 时：
1. 用 JD 前 80 个词构造 `refined_query`（原 JD 摘要）
2. 执行第二次 `rag_retrieve_tool`
3. 如果第二次检索成功且有结果，使用第二次的 chunks
4. 无论第二次结果如何，**不再继续重试**（bounded to 1 retry）

Trace 中记录：
- `retrieval_quality`（第一次检索质量）
- `query_refinement_used`（是否触发重试）
- `refinement_reason`（触发原因）
- `retrieval_attempts`（总检索次数，1 或 2）

## Model Provider Layer

参见 `llm_provider.py` 和 [架构文档](architecture.md) 中的 Provider Health Check 与 Fallback 部分。

当前支持的 provider：`gemini`、`deepseek`、`openai_compatible`、`mock`。

## Trace Layer

参见 [Trace 说明](../outputs/traces/) 和 `trace_utils.py`。

v2.5 新增 Trace 字段：
- `review_passed: bool | None` — Reviewer 审核结果
- `query_refinement_used: bool` — 是否使用了 query refinement
- `retrieval_attempts: int` — 检索总次数（1 或 2）

## Eval Layer

参见 [Eval 文档](eval.md)。

Eval Runner 使用 mock LLM 和 local embedding 独立运行，不依赖外部 API。当 Agent Workflow 中包含 Reviewer 步骤时，Eval 会自动验证 `review_result` 存在且结构正确。

## Fallback Layer

参见 [架构文档](architecture.md) Provider Health Check 与 Fallback 部分。

所有 provider（包括 deepseek）都支持 `fallback_to_mock`。真实模型失败时，`LLMResult.fallback_used = True`，Trace 中记录 `original_provider` 和 `provider_error`。

## Reviewer Agent

Reviewer Agent 是一个**纯规则审核器**，位于 Agent Workflow 的 LLM 分析步骤之后：

- 不调用真实 LLM，避免测试不稳定。
- 检查分析报告的结构完整性和证据使用情况。
- 输出直接进入 Trace 和页面展示。
- 如果 LLM 分析返回格式不完整（如 mock fallback 场景），Reviewer 会指出缺失部分。

## Bounded Loop

Query Refinement Loop 的边界设计：

| 属性 | 值 |
|------|-----|
| 最大重试次数 | 1 |
| 触发条件 | `evaluate_retrieval_quality` 返回 `"low"` |
| refined_query 构造 | JD 前 80 个词 |
| 失败行为 | 使用原始检索结果继续 |

## 当前边界

本项目 Harness 是轻量 MVP，明确不包含：

- 复杂 planning 或动态工具选择
- 长期记忆或持久化会话
- 多 Agent 协作
- 沙盒执行环境
- 生产级权限控制、限流或熔断
- LangChain / LangGraph 集成

这些边界是有意设定的，目的是保持代码可读、可讲解、可手动验证。
