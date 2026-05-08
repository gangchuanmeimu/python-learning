from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import config

# ---------------------- 初始化双模型 ----------------------
# 主模型：复杂推理、生成、校验用
main_llm = ChatOpenAI(
    model=config.LLM_MODEL_NAME,
    api_key=config.DASHSCOPE_API_KEY,
    base_url=config.DASHSCOPE_BASE_URL,
    temperature=0.3,
    max_retries=config.MAX_RETRY
)

# 轻量模型：简单分类、改写用，省Token速度快
lite_llm = ChatOpenAI(
    model=config.LITE_LLM_MODEL_NAME,
    api_key=config.DASHSCOPE_API_KEY,
    base_url=config.DASHSCOPE_BASE_URL,
    temperature=0.1,
    max_retries=config.MAX_RETRY
)

# ---------------------- Agent 1: 意图识别专家（新增） ----------------------
intent_recognition_prompt = ChatPromptTemplate.from_messages([
    ("system", f"你是专业的意图识别专家。你的任务是严格按照以下标签，对用户的问题进行意图分类，**只输出标签英文全称，不要任何其他内容**。\n\n标签列表：\n{config.INTENT_LABELS}\n\n分类规则：\n1. 询问知识库内相关内容 → KNOWLEDGE_QA\n2. 日常闲聊、打招呼、无意义对话 → CHITCHAT\n3. 询问的内容和知识库完全无关 → OUT_OF_SCOPE\n4. 敏感、违规、违法、政治相关问题 → SENSITIVE"),
    MessagesPlaceholder(variable_name="messages"),
])
intent_recognition_agent = intent_recognition_prompt | lite_llm

# ---------------------- Agent 2: Query改写专家（升级支持多轮上下文） ----------------------
query_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业的检索Query改写专家。用户的问题可能口语化、不完整、依赖上文对话上下文，你的任务是：\n1. 结合历史对话上下文，把用户的当前问题补全为完整、独立、无上下文依赖的问句\n2. 改写成适合知识库检索的专业、精准的问句\n3. **只输出改写后的Query，不要任何其他内容、解释、说明**。\n\n规则：\n- 保留用户问题的核心语义，不添加额外信息\n- 补全指代内容（如'他''它''这个'对应的具体主体）\n- 确保改写后的问句，单独拿出来也能明确知道用户要查什么"),
    MessagesPlaceholder(variable_name="messages"),
])
query_rewrite_agent = query_rewrite_prompt | lite_llm

# ---------------------- Agent 3: 路由决策专家（新增） ----------------------
router_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业的路由决策专家。你的任务是根据用户的意图分类，给出后续的执行路径，**只输出路径名称，不要任何其他内容**。\n\n路径规则：\n1. 意图为KNOWLEDGE_QA → 输出 RETRIEVE\n2. 意图为CHITCHAT → 输出 CHITCHAT_REPLY\n3. 意图为OUT_OF_SCOPE → 输出 REJECT\n4. 意图为SENSITIVE → 输出 REJECT"),
    ("human", "用户意图分类：{intent}"),
])
router_agent = router_prompt | lite_llm

# ---------------------- Agent 4: 闲聊回复专家（新增） ----------------------
chitchat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好、简洁的闲聊助手。请用简短、亲切的语气回复用户的闲聊内容，不要超过2句话，不要回答和知识库相关的内容。"),
    MessagesPlaceholder(variable_name="messages"),
])
chitchat_agent = chitchat_prompt | lite_llm

# ---------------------- Agent 5: 文档重排序专家（新增） ----------------------
rerank_prompt = ChatPromptTemplate.from_messages([
    ("system", f"你是专业的文档重排序专家。你的任务是根据用户的问题，对检索到的文档进行相关性打分排序，只保留最相关的{config.TOP_K_RERANK}篇，**只输出保留的文档内容，不要任何其他解释**。\n\n打分规则：\n1. 完全匹配用户问题核心 → 最高分\n2. 部分匹配相关知识点 → 中等分\n3. 无关内容 → 0分，直接剔除"),
    ("human", "用户问题：{query}\n\n检索到的文档：\n{retrieved_docs}"),
])
rerank_agent = rerank_prompt | main_llm

# ---------------------- Agent 6: 答案生成专家（优化原有） ----------------------
answer_generator_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的知识问答专家。你必须**100%仅基于以下提供的参考资料**来回答用户的问题，绝对禁止编造资料里没有的内容。\n\n如果参考资料中没有相关信息，请明确告诉用户“知识库中暂未找到相关内容”。\n\n回答要求：逻辑清晰、重点突出、准确专业，分点说明复杂内容。\n\n参考资料：\n{context}"),
    MessagesPlaceholder(variable_name="messages"),
])
answer_generator_agent = answer_generator_prompt | main_llm

# ---------------------- Agent 7: 事实校验&幻觉消除专家（新增） ----------------------
fact_check_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业的事实校验专家。你的任务是校验生成的答案是否完全基于参考资料，**绝对禁止编造内容、出现幻觉**。\n\n校验规则：\n1. 如果答案完全符合参考资料，直接输出原答案\n2. 如果答案有编造的内容，直接删除虚假部分，输出修正后的答案\n3. 如果答案完全脱离参考资料，直接输出“知识库中暂未找到相关内容”\n\n**只输出最终的校验后答案，不要任何校验过程、解释说明**。"),
    ("human", "参考资料：\n{context}\n\n生成的答案：{raw_answer}"),
])
fact_check_agent = fact_check_prompt | main_llm