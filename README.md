# AI Agent 简历与岗位匹配分析助手

## 项目简介

本项目是一个基于大语言模型的简历与岗位匹配分析原型。用户输入岗位描述和个人经历后，系统会生成岗位要求分析、个人能力分析、匹配度分析和简历优化建议。

项目目标是通过一个轻量级 AI Agent 原型，实践 Prompt Engineering、Agent Workflow 和大模型应用开发的基本流程。

## 技术栈

- Python
- Streamlit
- Gemini API
- Prompt Engineering
- Agent Workflow

## 核心功能

- 岗位要求分析
- 个人能力分析
- 岗位与简历匹配度分析
- 简历优化建议生成
- 短期学习计划生成
- 可选 RAG 检索增强分析
- RAG 支持本地 hash embedding fallback，避免演示时被 Gemini Embedding 免费额度限制卡住
- RAG 分析完成后会显示检索片段预览，便于查看系统基于哪些简历内容生成分析结果
- RAG 支持 local_bge 本地语义 embedding，提高检索质量

## RAG 向量模式

RAG 模式会先把简历切分成片段，写入本地 ChromaDB，再根据岗位描述召回相关片段生成分析。

最稳定的本地 hash embedding：

```env
EMBEDDING_PROVIDER=local
```

`local` 不调用外部 API，稳定、不需要下载模型，但语义能力较弱，更适合兜底和快速演示。

更接近真实 RAG 的本地语义 embedding：

```env
EMBEDDING_PROVIDER=local_bge
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

`local_bge` 使用 `sentence-transformers` 加载本地语义模型生成向量，相比 hash embedding 更能理解语义相似度。首次运行可能需要下载模型；下载完成后会使用本地缓存。如果模型下载、加载或推理失败，系统会自动 fallback 到 `local` hash embedding，不会让页面崩溃。

正式场景可以切换为 Gemini Embedding：

```env
EMBEDDING_PROVIDER=gemini
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

Gemini Embedding 语义效果较好，但会消耗 API 额度，免费层级可能触发 `429 RESOURCE_EXHAUSTED`；项目会自动 fallback 到本地 hash embedding。

普通分析模式仍然只调用 Gemini 文本生成模型，不受 RAG 向量模式影响。

RAG 分析完成后，页面会提供“查看 RAG 检索片段”展开区，展示本次召回的简历片段、来源文件名、chunk_index 和 distance 信息，方便核对模型依据。

## 项目结构

```text
resume-agent/
├─ app.py
├─ agent.py
├─ prompts.py
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ README.md
└─ screenshots/
```

新增 txt、pdf、docx 简历文件上传与文本解析功能，支持用户上传简历后直接进行岗位匹配分析。
