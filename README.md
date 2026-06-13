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
├─ rag.py                         # 文本切分、embedding、ChromaDB 入库和检索
├─ prompts.py                     # 系统 Prompt、普通分析 Prompt、RAG Prompt
├─ requirements.txt               # Python 依赖
├─ .env.example                   # 环境变量示例，不包含真实 API Key
├─ .gitignore                     # 忽略 .env、.venv、chroma_db 等本地文件
├─ simple_test.py                 # 最小测试脚本
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
