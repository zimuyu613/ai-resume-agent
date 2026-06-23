# Eval Runner 说明

## 目标

`eval_runner.py` 的目标是验证 AI Resume Agent 的工程链路能否稳定运行，包括 RAG 召回、轻量 Rerank、Agent Workflow、Trace 和输出结构。它不是自动判断回答好坏的模型裁判。

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
- `expected.json`：预期 section、关键词和输出标题。

## 当前评测指标

- RAG 工具是否成功。
- 是否返回至少一个 chunk。
- 是否命中预期 section。
- 召回文本是否覆盖部分预期关键词。
- Rerank 是否返回分数和关键词命中。
- Agent Workflow 是否返回分析与步骤。
- Trace 是否包含 run_id、步骤和 rerank 状态。
- 非 RAG 路径是否正确记录跳过检索。
- 输出是否包含预期标题或关键词。

结果保存在 `eval_results/eval_result_<timestamp>.json`，该目录属于本地运行产物，不提交 Git。

## Mock LLM 与真实 Gemini 的区别

Eval Runner 默认使用确定性 mock LLM。相同输入会得到固定结构，避免 API Key、网络、额度和模型随机性影响基础回归。

真实 Gemini 会产生更丰富但有波动的内容，也可能受到模型版本、Prompt、温度、地区和限流影响。mock 通过只能说明调用链路与结构正常，不能说明真实回答质量优秀。

## 为什么是工程型评测

当前 case 数量少，没有人工评分标准、参考答案质量等级、统计显著性或事实一致性判定。关键词和标题命中也只是结构检查。因此当前 Eval 适合做本地回归和面试展示，不应表述为严格模型质量评测。

## 后续升级方向

1. 扩充脱敏 case，覆盖不同岗位、简历长度和缺失信息场景。
2. 为检索增加 Recall@K、MRR、nDCG 等指标和人工相关性标注。
3. 对真实模型输出建立事实性、完整性、可执行性评分规则。
4. 使用人工双评或经过验证的 LLM-as-a-Judge，并记录评审版本。
5. 固定 Prompt、模型版本和随机参数，比较不同版本回归结果。
6. 增加失败 case、错误注入和性能基线。
