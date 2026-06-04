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

## RAG 向量模式

RAG 模式会先把简历切分成片段，写入本地 ChromaDB，再根据岗位描述召回相关片段生成分析。

默认使用本地 hash embedding：

```env
EMBEDDING_PROVIDER=local
```

本地模式不调用 Gemini Embedding API，更适合免费额度有限时演示。Gemini Embedding 免费层级可能触发 `429 RESOURCE_EXHAUSTED`，项目会自动 fallback 到本地 hash embedding。

正式场景可以切换为 Gemini Embedding：

```env
EMBEDDING_PROVIDER=gemini
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

普通分析模式仍然只调用 Gemini 文本生成模型，不受 RAG 向量模式影响。

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
