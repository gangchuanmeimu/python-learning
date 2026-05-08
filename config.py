import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ---------------------- 路径配置 ----------------------
# 项目根目录
BASE_DIR = Path(__file__).parent
# 知识库文档目录
DATA_DIR = BASE_DIR / "data"
# 向量数据库持久化目录
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# ---------------------- 通义千问 LLM 配置 ----------------------
# 千问 OpenAI 兼容接口地址
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 千问 API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# 使用的模型（推荐 qwen-plus 性价比高，也可以用 qwen-turbo 更快）
LLM_MODEL_NAME = "qwen-plus"
# 千问 Embedding 模型（用于向量化文档）
EMBEDDING_MODEL_NAME = "text-embedding-v3"

# ---------------------- 多智能体配置 ----------------------
# 检索返回的文档数量
TOP_K_RETRIEVE = 3
# 重排序后保留的文档数量
TOP_K_RERANK = 3
# 最大重试次数
MAX_RETRY = 3
# 轻量模型名称
LITE_LLM_MODEL_NAME = "qwen-turbo"
# 意图分类标签
INTENT_LABELS = """
- KNOWLEDGE_QA: 知识库相关问答
- CHITCHAT: 日常闲聊
- OUT_OF_SCOPE: 超出范围
- SENSITIVE: 敏感内容
"""
# ---------------------- 多轮对话记忆配置 ----------------------
# 最大保留的对话轮数
MAX_HISTORY_ROUNDS = 10
# 记忆持久化路径
HISTORY_DIR = BASE_DIR / "chat_history"