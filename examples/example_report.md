# AI 简历与岗位匹配分析报告（脱敏示例）

> 示例岗位：AI 应用开发工程师  
> 示例候选人：候选人 A（演示数据，不包含真实个人隐私）

## 岗位要求分析

岗位重点关注 Python 大模型应用开发、RAG 检索链路、Prompt Engineering 和基础 Agent 工程能力。候选人还需要能够解释 embedding、chunking、向量数据库和工具调用的实现过程，并具备基础测试与问题排查意识。

## 能力匹配

- 掌握 Python、Streamlit、ChromaDB 和 Gemini API 的基础使用。
- 有完整的 RAG Demo 经历，覆盖文档解析、section-aware chunking、embedding、top_k 检索和结果展示。
- 项目包含 Tool Calling、固定 Agent Workflow、Trace JSON 和 Eval Runner，能够展示 AI 应用工程化思路。
- 能够说明 local embedding fallback 与 API 限流场景下的稳定性处理。

## 差距分析

- 当前项目主要是本地 MVP，尚未提供独立后端服务与生产部署方案。
- RAG 检索仍缺少 rerank 和更严格的离线质量指标。
- Agent Workflow 使用固定工具链，没有复杂 planning、多 Agent 协作和长期记忆。
- Eval Runner 主要验证工程链路，尚不能代表真实 LLM 的回答质量。

## 简历优化建议

1. 在项目经历中补充可量化信息，例如样例数量、平均响应耗时和召回命中情况。
2. 明确区分普通 LLM、RAG 和 Agent Workflow 三种模式的输入与处理链路。
3. 用一条项目描述说明 Trace 如何帮助定位检索失败、工具失败或 embedding fallback。
4. 在面试中主动说明项目边界，并给出 rerank、FastAPI 和多模型支持的后续方案。

## 总结

候选人的经历与 AI 应用开发岗位具备基础匹配度，优势在于能够展示从 RAG、工具封装到 Trace 和 Eval 的完整 MVP 链路。后续应重点补充真实数据指标、检索质量评估和服务化经验。
