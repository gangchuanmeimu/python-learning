from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
import operator
from vector_store import build_vector_store
from agents import (
    intent_recognition_agent,
    query_rewrite_agent,
    router_agent,
    chitchat_agent,
    rerank_agent,
    answer_generator_agent,
    fact_check_agent
)
import config

# ---------------------- 1. 定义全局状态State（全链路数据传递） ----------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # 对话历史
    intent: str                                                 # 识别出的用户意图
    router_path: str                                            # 路由决策的路径
    rewritten_query: str                                        # 改写后的检索Query
    retrieved_docs: str                                         # 初次检索到的文档
    reranked_docs: str                                          # 重排序后的最终参考资料
    raw_answer: str                                             # 初次生成的答案
    final_answer: str                                           # 校验后的最终答案

# ---------------------- 2. 加载向量检索器 ----------------------
_, retriever = build_vector_store()

# ---------------------- 3. 定义工作流全节点 ----------------------
# 节点1：意图识别
def node_intent_recognition(state: AgentState):
    print("🤖 【1/7】正在调用【意图识别Agent】...")
    response = intent_recognition_agent.invoke({"messages": state["messages"]})
    intent = response.content.strip()
    print(f"   识别到用户意图：{config.INTENT_LABELS.get(intent, intent)}")
    return {"intent": intent}

# 节点2：路由决策
def node_router(state: AgentState):
    print("🤖 【2/7】正在调用【路由决策Agent】...")
    response = router_agent.invoke({"intent": state["intent"]})
    router_path = response.content.strip()
    print(f"   决策执行路径：{router_path}")
    return {"router_path": router_path}

# 节点3：Query改写
def node_query_rewrite(state: AgentState):
    print("🤖 【3/7】正在调用【Query改写Agent】...")
    response = query_rewrite_agent.invoke({"messages": state["messages"]})
    rewritten_query = response.content.strip()
    print(f"   改写后的检索Query：{rewritten_query}")
    return {"rewritten_query": rewritten_query}

# 节点4：知识库检索
def node_retrieve(state: AgentState):
    print("🔍 【4/7】正在调用【知识库检索Agent】...")
    query = state["rewritten_query"]
    docs = retriever.invoke(query)
    context = "\n\n".join([f"[资料 {i+1}]:\n{doc.page_content}" for i, doc in enumerate(docs)])
    print(f"   初次检索到 {len(docs)} 条相关资料")
    return {"retrieved_docs": context}

# 节点5：文档重排序
def node_rerank(state: AgentState):
    print("📊 【5/7】正在调用【文档重排序Agent】...")
    response = rerank_agent.invoke({
        "query": state["rewritten_query"],
        "retrieved_docs": state["retrieved_docs"]
    })
    reranked_docs = response.content.strip()
    print(f"   重排序完成，已筛选最相关的{config.TOP_K_RERANK}条资料")
    return {"reranked_docs": reranked_docs}

# 节点6：答案生成
def node_generate_answer(state: AgentState):
    print("✍️  【6/7】正在调用【答案生成Agent】...")
    response = answer_generator_agent.invoke({
        "messages": state["messages"],
        "context": state["reranked_docs"]
    })
    raw_answer = response.content.strip()
    return {"raw_answer": raw_answer}

# 节点7：事实校验&幻觉消除
def node_fact_check(state: AgentState):
    print("✅ 【7/7】正在调用【事实校验Agent】...")
    response = fact_check_agent.invoke({
        "context": state["reranked_docs"],
        "raw_answer": state["raw_answer"]
    })
    final_answer = response.content.strip()
    return {"final_answer": final_answer, "messages": [HumanMessage(content=final_answer)]}

# 分支节点A：闲聊回复
def node_chitchat_reply(state: AgentState):
    print("💬 正在调用【闲聊回复Agent】...")
    response = chitchat_agent.invoke({"messages": state["messages"]})
    final_answer = response.content.strip()
    return {"final_answer": final_answer, "messages": [HumanMessage(content=final_answer)]}

# 分支节点B：拒绝回复
def node_reject_reply(state: AgentState):
    print("🚫 正在调用【拒绝回复Agent】...")
    intent = state["intent"]
    if intent == "SENSITIVE":
        final_answer = "很抱歉，这个问题我无法回答，请你遵守相关法律法规，询问合规内容。"
    else:
        final_answer = "很抱歉，这个问题超出了我的知识库范围，我只能回答知识库内的相关内容。"
    return {"final_answer": final_answer, "messages": [HumanMessage(content=final_answer)]}

# ---------------------- 4. 路由条件判断函数 ----------------------
def router_branch(state: AgentState):
    """根据路由路径，决定下一步走哪个分支"""
    return state["router_path"]

# ---------------------- 5. 构建并编译LangGraph工作流 ----------------------
workflow = StateGraph(AgentState)

# 1. 注册所有节点
workflow.add_node("intent_recognition", node_intent_recognition)
workflow.add_node("router", node_router)
workflow.add_node("query_rewrite", node_query_rewrite)
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("rerank", node_rerank)
workflow.add_node("generate_answer", node_generate_answer)
workflow.add_node("fact_check", node_fact_check)
workflow.add_node("chitchat_reply", node_chitchat_reply)
workflow.add_node("reject_reply", node_reject_reply)

# 2. 定义执行顺序（主链路）
workflow.set_entry_point("intent_recognition")  # 入口：先做意图识别
workflow.add_edge("intent_recognition", "router")  # 识别完做路由决策

# 3. 定义条件分支（核心！不同意图走不同流程）
workflow.add_conditional_edges(
    "router",
    router_branch,
    {
        "RETRIEVE": "query_rewrite",        # 知识库问答：走检索全流程
        "CHITCHAT_REPLY": "chitchat_reply", # 闲聊：直接走闲聊回复
        "REJECT": "reject_reply"             # 超出范围/敏感：直接拒绝
    }
)

# 4. 知识库问答的完整链路
workflow.add_edge("query_rewrite", "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate_answer")
workflow.add_edge("generate_answer", "fact_check")

# 5. 所有分支最终都指向结束
workflow.add_edge("fact_check", END)
workflow.add_edge("chitchat_reply", END)
workflow.add_edge("reject_reply", END)

# 6. 编译工作流（生成可调用的app）
app = workflow.compile()