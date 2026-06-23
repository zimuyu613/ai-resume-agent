# AI 简历与岗位匹配分析助手

## 项目简介

这是一个基于 Python、Streamlit、Gemini API 和 ChromaDB 的 RAG 简历与岗位匹配分析助手。用户可以上传简历或使用示例简历，输入岗位 JD 后，系统会生成岗位要求分析、个人能力分析、匹配度分析和简历优化建议。

当前项目是一个面向学习和展示的 AI 应用工程化 MVP，更准确地说是一个 “LLM + RAG Workflow” 原型，不是完整生产级多工具 Agent。

## 核心功能

- 支持 txt / pdf / docx 简历解析
- 支持岗位 JD 输入
- 支持普通 LLM 分析模式
- 支持 RAG 检索增强分析模式
- 支持 local / local_bge / gemini 三种 embedding provider
- 支持 ChromaDB 本地向量检索
- 支持 RAG 召回片段、来源、chunk_index、distance 展示
- 支持示例简历和示例岗位 JD 一键加载
- 支持 Markdown 分析报告导出
- 对 Gemini API 常见错误提供友好提示

## 技术栈

- Python
- Streamlit
- Gemini API
- ChromaDB
- sentence-transformers
- pypdf
- python-docx
- python-dotenv
- local hash embedding fallback

## 项目结构

```text
resume-agent/
├─ app.py                         # Streamlit 页面入口，负责上传、输入、展示和导出
├─ agent.py                       # 普通分析和 RAG 分析工作流，负责调用大模型
├─ agent_workflow.py              # Tool Calling MVP 的固定 Agent Workflow 编排
├─ tools.py                       # 轻量级工具层和统一 ToolResult 返回结构
├─ trace_utils.py                 # Trace 数据结构、摘要、序列化和 JSON 保存
├─ eval_runner.py                 # 工程型基础评测入口
├─ rag.py                         # 文本切分、embedding、ChromaDB 入库和检索
├─ prompts.py                     # 系统 Prompt、普通分析 Prompt、RAG Prompt
├─ requirements.txt               # Python 依赖
├─ .env.example                   # 环境变量示例，不包含真实 API Key
├─ .gitignore                     # 忽略 .env、.venv、chroma_db 等本地文件
├─ simple_test.py                 # 最小测试脚本
├─ eval_cases/                    # 可提交的脱敏评测样例、JD 和 expected.json
├─ samples/
│  ├─ sample_resume.txt           # 脱敏示例简历
│  └─ sample_job_description.txt  # 示例岗位 JD
└─ README.md
```

## 安装与运行

Windows PowerShell 示例：

```powershell
cd D:\AIProjects\resume-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

如果 PowerShell 不允许激活脚本，可以直接使用虚拟环境里的 Python：

```powershell
cd D:\AIProjects\resume-agent
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器打开终端显示的地址，通常是：

```text
http://localhost:8501
```

### 方式一：命令行启动

```powershell
cd D:\AIProjects\resume-agent
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### 方式二：双击启动

在项目根目录双击：

```text
run_app.bat
```

如果窗口提示未找到 `.venv\Scripts\python.exe`，请先创建虚拟环境并安装依赖。

## .env 配置

复制 `.env.example` 为 `.env`，并填写自己的 Gemini API Key：

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# 可选：local / local_bge / gemini
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

不要提交真实 `.env` 或 API Key。

### Embedding Provider 说明

- `local`：本地 hash embedding，不调用外部 API，最稳定，但语义能力较弱。
- `local_bge`：使用 sentence-transformers 本地语义模型，检索质量更接近真实 RAG；首次运行可能下载模型。
- `gemini`：使用 Gemini Embedding，语义能力较好，但会消耗 API 额度，可能触发 429。

如果 `local_bge` 或 `gemini` 失败，系统会自动 fallback 到 `local` hash embedding，避免 RAG 页面崩溃。

## 使用方式

### 方式一：上传自己的简历

1. 在页面中输入岗位描述。
2. 上传 txt、pdf 或 docx 简历文件。
3. 选择是否勾选 `启用 RAG 检索增强分析`。
4. 点击 `开始分析`。
5. 查看四个分析 Tab。
6. 如使用 RAG，可展开 `查看 RAG 检索片段`。
7. 点击 `下载分析报告 Markdown` 导出报告。

### 方式二：加载示例数据

1. 点击 `加载示例简历和岗位 JD`。
2. 页面会自动填入示例简历和示例岗位 JD。
3. 可直接点击 `开始分析`。
4. 可勾选 RAG 模式查看召回片段。

## RAG 流程说明

RAG 模式会执行以下步骤：

1. 读取简历文本。
2. 将简历切分为 chunk。
3. 使用 local、local_bge 或 gemini 生成 embedding。
4. 将 chunk 和 metadata 写入 ChromaDB。
5. 使用岗位 JD 检索 top_k 相关简历片段。
6. 将检索片段拼入 RAG Prompt。
7. 调用 Gemini 生成结构化分析结果。
8. 在页面展示分析结果和召回片段来源。

当前默认 top_k 为 3，并限制最大处理 chunk 数，避免免费 API 或长文本导致不稳定。

当前项目采用 section-aware chunking：先识别简历模块，例如 `basic_info`、`skills`、`project_experience`，再在每个模块内部切分 chunk。这样可以减少 chunk 跨模块导致的 section 标注不准确问题。section 识别仍然是基于规则的方法，不是复杂 NLP。

### RAG 参数与可解释性

- chunk：简历文本切分后的片段，是写入向量数据库和检索召回的基本单位。
- overlap：相邻 chunk 的重叠区域，用于减少关键信息刚好被切断的问题。
- top_k：从 ChromaDB 中召回最相关的前 k 个片段。页面中可以在 RAG 模式下调整，默认值为 3。
- metadata：每个 chunk 会保存来源和位置等信息，例如 `source`、`chunk_id`、`file_name`、`char_start`、`char_end`、`chunk_length`。这些信息用于解释模型参考了哪些简历内容。
- section metadata：项目会用简单规则识别简历模块，例如教育背景、技能栈、项目经历、实习经历、获奖竞赛和自我评价。RAG 模式可以按 section 缩小检索范围。
- distance：ChromaDB 返回的检索距离，数值越小通常表示越相关。不同 embedding 模式下 distance 的绝对值不一定可直接横向比较，更适合作为同一次检索中的参考。

当前支持的 section 类型：

- `education`：教育背景 / 教育经历 / Education
- `basic_info`：姓名 / 求职意向 / 个人信息 / 基本信息 / 联系方式 / 邮箱 / 手机 / 意向岗位
- `skills`：专业技能 / 技能栈 / 技术栈 / Skills
- `project_experience`：项目经历 / 项目经验 / Projects
- `internship_experience`：实习经历 / 工作经历 / Internship / Work Experience
- `awards`：竞赛经历 / 获奖经历 / 荣誉奖项 / Awards
- `self_evaluation`：自我评价 / 个人总结 / Summary
- `unknown`：无法识别的内容

section 识别是规则方法，不是复杂 NLP。如果某个 section 没有召回结果，可以切回“全部”检索。

当前 RAG 仍有一些不足：

- 还没有 rerank，召回结果完全依赖向量相似度。
- 还没有系统化检索质量评估。
- metadata 过滤还比较基础，暂未按教育背景、项目经历、技能等 section 精细过滤。
- local hash embedding 语义能力有限，更适合作为稳定兜底。

## Agent Workflow / Tool Calling MVP

本版本新增了一个轻量级 Tool Calling / Agent Tool Layer，让项目在保持 RAG Workflow MVP 简洁性的基础上，更接近 AI Agent 应用开发岗位常见的表达方式。

新增工具层位于 `tools.py`，统一使用 `ToolResult` 返回结构：

- `parse_resume_tool`：封装 txt / pdf / docx 或直接文本输入的简历解析。
- `rag_retrieve_tool`：复用现有 RAG 检索流程，返回 chunks、section、chunk_index、distance、chunk_length 等可解释信息。
- `llm_match_analysis_tool`：复用现有 Gemini 分析逻辑，可执行普通 LLM 分析或 RAG 增强分析。
- `export_markdown_tool`：将分析结果和 metadata 组装为 Markdown，也可以保存到本地文件。

新增固定流程编排位于 `agent_workflow.py`：

1. 接收简历文本和岗位 JD。
2. 根据 `use_rag=True` 调用 `rag_retrieve_tool`。
3. 将召回片段传入 `llm_match_analysis_tool`。
4. 返回最终分析结果，并在 `workflow_steps` 中记录每一步的 `step_name`、`tool_name`、`success`、`message` 和简要数据摘要。

Streamlit 页面新增了“Agent Workflow 分析”模式。选择该模式后，页面会展示最终分析结果、Agent Workflow Steps，以及 RAG 召回片段的 section、chunk_index、distance、chunk_length 和内容预览。

当前版本仍不是完整生产级 Agent：没有复杂 planning、多 Agent 协作、长期记忆、权限系统或生产级观测平台。它的目标是作为面试展示中的 Tool Calling MVP，说明项目已经具备“工具封装 -> 固定工具链编排 -> 页面可解释展示”的 Agent 化雏形。

## Trace / Observability MVP

项目新增了轻量级 trace 能力。每次 Agent Workflow 运行会生成独立 `run_id`，并记录总耗时、简历和岗位 JD 输入长度、`top_k`、embedding provider、是否使用 RAG、是否发生 embedding fallback，以及最终运行状态。

每个工具步骤会记录：

- `step_name` 和 `tool_name`
- 工具调用成功状态、消息和错误信息
- 脱敏后的输入摘要与输出摘要
- 开始时间、结束时间和单步耗时

页面可以展开查看 RAG 检索和 LLM 分析步骤，用于排查 RAG 召回数量、主要 section、distance 范围、工具调用状态和 LLM 是否产生分析结果。Trace 可以直接通过 `st.json` 查看，也可以下载为 `trace_<run_id>.json`。

Agent Workflow 完成后还会尝试把 trace 保存到 `outputs/traces/`。保存失败不会中断分析，页面会显示友好提示。

当前 trace 是教学和面试展示级别的轻量实现，不等同于 LangSmith、OpenTelemetry 等生产级观测平台，也不包含分布式链路追踪、集中日志检索、告警和权限治理。

## Eval Runner MVP

项目新增了轻量级基础评测机制。`eval_cases/` 中包含脱敏示例简历、岗位 JD 和对应的 `expected.json`，当前样例覆盖 AI 应用开发工程师和 Python RAG 实习生场景。

`eval_runner.py` 会逐个检查：

- RAG 工具是否成功返回片段。
- 召回结果是否覆盖预期 section 和部分关键词。
- Agent Workflow 是否返回分析结果和工具步骤。
- Trace 是否包含 run_id 和步骤记录。
- 非 RAG Workflow 是否正确跳过并记录 RAG 步骤。
- 分析输出是否包含预期标题或关键词。

运行命令：

```powershell
cd D:\AIProjects\resume-agent
.\.venv\Scripts\python.exe eval_runner.py
```

评测结果会输出到控制台，并保存到 `eval_results/eval_result_<timestamp>.json`。运行生成的 eval 结果不会提交到 Git，`eval_cases/` 示例数据则应正常提交。

为了避免依赖 Gemini API Key、网络状态和免费额度，当前 Eval 默认使用 local embedding 与确定性的 mock LLM，并在结果中标记 `llm_mode: "mock"`。因此它是工程型基础评测，用于验证流程稳定性、RAG 召回是否覆盖关键 section、Agent Workflow 和 Trace 是否正常工作；它不是严格的模型回答质量评测，也不代表真实 LLM 的效果分数。

## 运行测试

项目提供了一个最小测试脚本：

```powershell
cd D:\AIProjects\resume-agent
.\.venv\Scripts\python.exe simple_test.py
```

测试内容包括：

- `split_text` 能正常切分文本
- 空文本不会崩溃
- local embedding 返回固定长度向量
- 示例简历和示例 JD 存在并可读取
- `ToolResult` 结构可以正常创建
- `rag_retrieve_tool` 在简单输入下可以完成基础召回
- `run_resume_agent_workflow` 可以在 mock LLM 下返回 `workflow_steps`
- `TraceStep` / `WorkflowTrace` 可以创建并序列化为 JSON
- Agent Workflow 返回 run_id、steps、耗时和开始/结束时间
- `eval_cases` 样例目录结构完整，Eval Runner 核心函数可以导入

基础测试不会强制真实调用 Gemini API，因此没有 API Key 时也可以验证本地 RAG 和 Tool Calling MVP 的主要结构。

## 常见问题

### 1. Gemini 429 / RESOURCE_EXHAUSTED 怎么办？

这说明 Gemini API 免费额度不足或请求过多。可以稍后重试、降低点击频率、更换 API Key，或开通更高配额。注意：即使 RAG embedding 使用 local，最终报告生成仍会调用 Gemini 文本生成模型。

### 2. User location is not supported 怎么办？

这通常表示当前网络出口地区不支持 Gemini API。可以检查网络环境，或切换到其他可用的大模型服务。

### 3. 模型名错误怎么办？

检查 `.env` 中的 `GEMINI_MODEL`，建议使用类似：

```env
GEMINI_MODEL=gemini-2.5-flash
```

不要写成 `models/gemini-2.5-flash` 或带空格的模型名。

### 4. ChromaDB 依赖问题怎么办？

先确认已经安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

项目已经忽略 `chroma_db/`，本地向量数据库不会提交到 Git。

### 5. local embedding 和 Gemini embedding 有什么区别？

- local hash embedding：稳定、本地运行、无需 API，但语义能力有限。
- local_bge：本地语义模型，检索质量更好，首次可能下载模型。
- Gemini embedding：云端语义模型，效果较好，但消耗 API 额度。

## 当前不足与后续计划

当前项目仍是 RAG Workflow 原型，不是完整生产级 Agent。后续可以继续优化：

- 增加 metadata 过滤和更细粒度的 section 标记
- 增加 rerank，提高召回质量
- 增加 FastAPI 服务化接口
- 增加基础单元测试和端到端测试
- 增加 Trace / 日志面板
- 增加 Tool Calling 示例
- 增加更多模型 provider，降低 Gemini 单点依赖
