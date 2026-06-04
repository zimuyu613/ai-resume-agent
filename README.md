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

新增 txt、pdf、docx 简历文件上传与文本解析功能，支持用户上传简历后直接进行岗位匹配分析。