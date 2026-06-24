# Eval Runner 说明

## 目标

`eval_runner.py` 的目标是评估 RAG 召回质量，并继续验证 Agent Workflow、Trace 和输出结构能否稳定运行。它用于比较普通 RAG 与 RAG + Rerank，不是自动判断最终回答好坏的模型裁判。

## Eval Case 结构

```text
eval_cases/
└─ case_name/
   ├─ resume.txt
   ├─ jd.txt
   └─ expected.json
```

- `resume.txt`：脱敏简历样例。
- `jd.txt`：目标岗位描述。
- `expected.json`：预期 section、关键词、输出标题和简化 gold evidence。

Gold evidence 使用 `section + keywords` 表示，例如技能模块中应出现 Python 或 RAG。它不绑定 chunk_id，因此切分规则轻微变化后仍可使用。

## 当前评测指标

- retrieved count、section hit rate 和 keyword hit rate。
- gold recall：所有 gold evidence 中有多少被召回。
- Recall@1、Recall@3、Recall@5。
- MRR。
- 普通 RAG 与 RAG + Rerank 的 improved / same / worse 对比。
- Rerank 是否返回分数和关键词命中。
- Agent Workflow 是否返回分析与步骤。
- Trace 是否包含 run_id、步骤和 rerank 状态。
- 非 RAG 路径是否正确记录跳过检索。
- 输出是否包含预期标题或关键词。

结果保存在 `eval_results/eval_result_<timestamp>.json`，同时生成 `eval_summary_<timestamp>.md`。该目录属于本地运行产物，不提交 Git。

## Recall@K 是什么

Recall@K 表示只看排名最前面的 K 个 chunk 时，能够覆盖多少 gold evidence。例如有两个 gold evidence，前三个 chunk 命中其中一个，则 Recall@3 为 0.5。它回答的是“有限上下文窗口里召回得全不全”。

## MRR 是什么

MRR 关注第一个相关 chunk 出现得有多早。如果第一个相关 chunk 排名第 1，得分是 1；排名第 2，得分是 1/2；完全没有相关 chunk，则为 0。当前项目按每个 case 的第一个 gold evidence 命中位置计算简化 MRR。

## 为什么比较 RAG 和 RAG + Rerank

向量检索先提供候选片段，rule-based Rerank 再根据 JD 关键词、section 和 distance 调整顺序。对比 Recall@3 与 MRR，可以观察 Rerank 是否把相关片段提前，还是保持不变或造成下降。

## Mock LLM 与真实 Gemini 的区别

Eval Runner 默认使用确定性 mock LLM。相同输入会得到固定结构，避免 API Key、网络、额度和模型随机性影响基础回归。

当前 Eval 通过统一 `llm_provider` 的 mock 路径运行，而不是直接调用 Gemini 或 OpenAI-compatible 服务。评测重点仍是 RAG 召回、Rerank 和工程链路。

Eval 和 API smoke test 都显式使用 Mock Provider，因此不依赖真实模型健康状态。Fallback 相关测试只验证错误捕获、元数据和工程链路能够继续运行，不代表 Gemini 或兼容模型的输出质量。

真实 Gemini 会产生更丰富但有波动的内容，也可能受到模型版本、Prompt、温度、地区和限流影响。mock 通过只能说明调用链路与结构正常，不能说明真实回答质量优秀。

## 为什么是工程型评测

当前评测存在明确限制：

- 样例数量少，岗位和简历分布有限。
- Gold evidence 使用 section + keywords，不是人工逐 chunk 相关性标注。
- Mock LLM 只保证结构稳定，不代表真实 Gemini 输出质量。
- 主要评估检索排序和工程链路，不评估最终答案的事实性与建议质量。

因此当前 Eval 适合做本地回归和面试展示，不应表述为严格工业级质量评测。

## 后续升级方向

1. 扩充脱敏 case，覆盖不同岗位、简历长度和缺失信息场景。
2. 增加人工逐 chunk 相关性标注、nDCG 和分岗位统计。
3. 对真实模型输出建立事实性、完整性、可执行性评分规则。
4. 使用人工双评或经过验证的 LLM-as-a-Judge，并记录评审版本。
5. 固定 Prompt、模型版本和随机参数，比较不同版本回归结果。
6. 增加失败 case、错误注入和性能基线。
7. 在固定数据、Prompt 和采样参数后，对不同真实 LLM 的输出质量做独立对比。
