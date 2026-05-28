import streamlit as st
from agent import run_agent_workflow

st.set_page_config(
    page_title="AI Agent 简历与岗位匹配分析助手",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Agent 简历与岗位匹配分析助手")

st.write(
    "输入岗位描述和个人经历，系统会分步骤分析岗位要求、个人匹配点、能力差距和简历优化建议。"
)

st.info("提示：当前 Demo 为原型版本，建议输入精简后的岗位描述和个人经历，以保证分析稳定性。")

with st.sidebar:
    st.header("项目说明")
    st.write("这是一个用于学习 AI Agent Workflow 的小型原型项目。")
    st.write("当前版本包含：")
    st.write("- 岗位要求提取")
    st.write("- 简历能力分析")
    st.write("- 匹配度分析")
    st.write("- 简历优化建议")
    st.write("")
    st.write("后续可扩展：")
    st.write("- Tool Calling")
    st.write("- RAG 文档检索")
    st.write("- PDF / Word 简历上传")

job_description = st.text_area(
    "请输入岗位描述",
    height=260,
    placeholder="例如：粘贴 AI Agent 开发实习生的岗位描述。",
)

resume_text = st.text_area(
    "请输入个人简历或项目经历",
    height=260,
    placeholder="例如：粘贴你的专业技能、项目经历和自我评价内容。",
)

if st.button("开始分析"):
    if not job_description.strip() or not resume_text.strip():
        st.warning("请先输入岗位描述和个人经历。")
    else:
        with st.spinner("正在分析中，请稍等..."):
            result = run_agent_workflow(job_description, resume_text)

        st.success("分析完成")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["岗位要求分析", "个人能力分析", "匹配度分析", "简历优化建议"]
        )

        with tab1:
            st.markdown(result["job_analysis"])

        with tab2:
            st.markdown(result["resume_analysis"])

        with tab3:
            st.markdown(result["match_analysis"])

        with tab4:
            st.markdown(result["suggestions"])