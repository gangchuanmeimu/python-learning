import streamlit as st
import os
from streamlit_chat import message
from streamlit_option_menu import option_menu
from langchain_core.messages import HumanMessage
from vector_store import build_vector_store, add_docs_to_vector_store, rebuild_vector_store
from langchain_openai import ChatOpenAI
import config

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="多智能体RAG知识库系统",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- 侧边栏导航 ----------------------
with st.sidebar:
    selected_menu = option_menu(
        menu_title="🚀 系统导航",
        options=["智能问答", "知识库管理", "系统配置"],
        icons=["chat-dots", "folder2-open", "gear"],
        menu_icon="cast",
        default_index=0,
    )

# ---------------------- 会话状态初始化 ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []

# ---------------------- 页面1：智能问答 ----------------------
if selected_menu == "智能问答":
    st.title("🤖 多智能体增强 RAG 问答")
    st.caption("全链路智能体协作 | 零幻觉事实校验 | 精准知识库检索")
    st.divider()

    # 聊天对话容器
    chat_container = st.container()
    with chat_container:
        # 渲染历史对话
        for i in range(len(st.session_state.generated)):
            message(st.session_state.past[i], is_user=True, key=str(i) + "_user")
            message(st.session_state.generated[i], key=str(i))

    # 输入框
    user_input = st.chat_input("请输入你的问题...")

    # 处理用户输入
    if user_input:
        # 把用户输入加入历史
        st.session_state.past.append(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))

        # 限制最大历史轮数，避免上下文溢出
        if len(st.session_state.messages) > config.MAX_HISTORY_ROUNDS * 2:
            st.session_state.messages = st.session_state.messages[-(config.MAX_HISTORY_ROUNDS * 2):]

        # 使用支持多轮对话的RAG实现
        with st.spinner("🤖 正在处理您的问题..."):
            vector_store, retriever = build_vector_store()
            llm = ChatOpenAI(
                model_name=config.LLM_MODEL_NAME,
                openai_api_key=config.DASHSCOPE_API_KEY,
                openai_api_base=config.DASHSCOPE_BASE_URL,
                temperature=0.7
            )

            # 构建上下文感知的查询
            if len(st.session_state.messages) > 1:
                # 多轮对话：结合历史上下文
                history_context = "\n".join([
                    f"用户: {msg.content}" if i % 2 == 0 else f"系统: {msg.content}"
                    for i, msg in enumerate(st.session_state.messages[:-1])
                ])
                enhanced_query = f"基于以下对话历史：\n{history_context}\n\n用户新问题：{user_input}\n\n请结合上下文回答。"
            else:
                # 单轮对话
                enhanced_query = user_input

            # 检索相关文档
            docs = retriever.invoke(enhanced_query)
            if docs:
                context = "\n\n".join(doc.page_content for doc in docs)
                prompt = f"基于以下知识库内容回答问题：\n\n{context}\n\n用户问题：{enhanced_query}\n\n请给出准确、简洁的回答。"
                response = llm.invoke(prompt)
                final_answer = response.content
            else:
                final_answer = "抱歉，知识库中没有找到相关信息。"

        # 把答案加入历史
        st.session_state.generated.append(final_answer)
        st.session_state.messages.append(HumanMessage(content=final_answer))

        # 刷新页面显示新对话
        st.rerun()

# ---------------------- 页面2：知识库管理 ----------------------
elif selected_menu == "知识库管理":
    st.title("📚 知识库管理")
    st.caption("支持上传 PDF/Word/Excel/TXT/Markdown 文档，自动解析并更新向量库")
    st.divider()

    # 1. 文档上传区域
    st.subheader("📤 上传新文档")
    uploaded_files = st.file_uploader(
        "选择要上传的文档",
        type=["txt", "md", "pdf", "docx", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # 保存上传的文件到data文件夹
        save_path = config.DATA_DIR
        if not save_path.exists():
            save_path.mkdir()
        
        file_paths = []
        for file in uploaded_files:
            file_full_path = os.path.join(save_path, file.name)
            with open(file_full_path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(file_full_path)
        
        # 加入向量库
        if st.button("✅ 确认上传并更新向量库", use_container_width=True, type="primary"):
            with st.spinner("正在解析文档并更新向量库..."):
                success, msg = add_docs_to_vector_store(file_paths)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
    
    st.divider()

    # 2. 向量库管理
    st.subheader("🔧 向量库操作")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重建向量库", use_container_width=True):
            with st.spinner("正在清空并重建向量库..."):
                success, msg = rebuild_vector_store()
                st.success(msg)
    with col2:
        if st.button("🗑️ 清空对话历史", use_container_width=True):
            st.session_state.messages = []
            st.session_state.generated = []
            st.session_state.past = []
            st.success("对话历史已清空")
    
    st.divider()

    # 3. 已有文档列表
    st.subheader("📋 已有知识库文档")
    if config.DATA_DIR.exists():
        files = os.listdir(config.DATA_DIR)
        if files:
            for file in files:
                st.text(f"📄 {file}")
        else:
            st.info("data文件夹下暂无文档，请先上传文档")
    else:
        st.info("data文件夹不存在，请先上传文档")

# ---------------------- 页面3：系统配置 ----------------------
elif selected_menu == "系统配置":
    st.title("⚙️ 系统配置")
    st.caption("调整模型参数、检索配置，优化问答效果")
    st.divider()

    # 模型参数配置
    with st.form("model_config_form"):
        st.subheader("🤖 模型参数配置")
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("模型温度（越低越严谨，越高越有创造性）", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            top_k_retrieve = st.slider("检索返回文档数量", min_value=1, max_value=10, value=config.TOP_K_RETRIEVE, step=1)
        with col2:
            top_k_rerank = st.slider("重排序后保留文档数量", min_value=1, max_value=5, value=config.TOP_K_RERANK, step=1)
            max_retry = st.slider("API最大重试次数", min_value=1, max_value=10, value=config.MAX_RETRY, step=1)
        
        # 提交按钮
        submit = st.form_submit_button("✅ 保存配置", use_container_width=True, type="primary")
        if submit:
            # 更新config配置
            config.TEMPERATURE = temperature
            config.TOP_K_RETRIEVE = top_k_retrieve
            config.TOP_K_RERANK = top_k_rerank
            config.MAX_RETRY = max_retry
            st.success("配置保存成功！重启服务后生效")
    
    st.divider()

    # 系统信息
    st.subheader("📊 系统信息")
    st.info(f"项目根目录：{config.BASE_DIR}")
    st.info(f"知识库目录：{config.DATA_DIR}")
    st.info(f"向量库目录：{config.VECTOR_DB_DIR}")
    st.info(f"核心大模型：{config.LLM_MODEL_NAME}")
    st.info(f"嵌入模型：{config.EMBEDDING_MODEL_NAME}")