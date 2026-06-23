# Trace JSON 说明

Agent Workflow 每次运行都会生成一个轻量级 Trace，记录 `run_id`、输入长度、embedding provider、工具调用步骤、输入输出摘要、错误信息和耗时。

Trace 默认保存到：

```text
outputs/traces/trace_<run_id>.json
```

Streamlit 页面也会通过 `st.json` 展示 Trace，并提供 JSON 下载按钮。

`outputs/traces/` 属于本地运行产物，已被 `.gitignore` 忽略，因此仓库不提交真实运行 Trace，只保留本说明文件。Trace 内容虽然只保存摘要，实际使用时仍应避免在输入或日志中放入不必要的个人隐私。
