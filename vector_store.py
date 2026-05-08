from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import config
import os

# ---------------------- 支持的文档格式 ----------------------
SUPPORTED_FORMAT = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader
}

def build_vector_store():
    """构建或加载向量数据库"""
    # 1. 初始化千问 Embedding 模型
    embeddings = DashScopeEmbeddings(
        model=config.EMBEDDING_MODEL_NAME,
        dashscope_api_key=config.DASHSCOPE_API_KEY
    )

    # 2. 如果向量库已存在，直接加载
    if config.VECTOR_DB_DIR.exists() and any(config.VECTOR_DB_DIR.iterdir()):
        print("正在加载已有向量数据库...")
        vector_store = Chroma(
            persist_directory=str(config.VECTOR_DB_DIR),
            embedding_function=embeddings
        )
        return vector_store, vector_store.as_retriever(search_kwargs={"k": config.TOP_K_RETRIEVE})

    # 3. 如果向量库不存在，构建新的
    print("正在构建新的向量数据库...")
    
    # 3.1 加载 data 文件夹下所有支持的格式
    all_docs = []
    for root, _, files in os.walk(config.DATA_DIR):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in SUPPORTED_FORMAT:
                file_path = os.path.join(root, file)
                loader_cls = SUPPORTED_FORMAT[file_ext]
                loader = loader_cls(file_path)
                docs = loader.load()
                # 给每个文档添加来源信息
                for doc in docs:
                    doc.metadata["source"] = file
                all_docs.extend(docs)
    
    if not all_docs:
        raise ValueError("data 文件夹下没有找到支持的文档，请先添加文档！")
    print(f"成功加载 {len(all_docs)} 个文档，来自 {len(set([doc.metadata['source'] for doc in all_docs]))} 个文件")

    # 3.2 语义分块（比固定分块效果更好）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    split_docs = text_splitter.split_documents(all_docs)
    print(f"文档切分完成，共 {len(split_docs)} 个文本块")

    # 3.3 向量化并存入 ChromaDB
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(config.VECTOR_DB_DIR)
    )
    
    print("向量数据库构建完成！")
    return vector_store, vector_store.as_retriever(search_kwargs={"k": config.TOP_K_RETRIEVE})

def add_docs_to_vector_store(file_paths: list):
    """新增文档到已有向量库"""
    # 加载已有向量库
    embeddings = DashScopeEmbeddings(
        model=config.EMBEDDING_MODEL_NAME,
        dashscope_api_key=config.DASHSCOPE_API_KEY
    )
    vector_store = Chroma(
        persist_directory=str(config.VECTOR_DB_DIR),
        embedding_function=embeddings
    )

    # 加载新文档
    all_docs = []
    for file_path in file_paths:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in SUPPORTED_FORMAT:
            loader_cls = SUPPORTED_FORMAT[file_ext]
            loader = loader_cls(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)
            all_docs.extend(docs)
    
    if not all_docs:
        return False, "没有支持的文档格式"
    
    # 分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    split_docs = text_splitter.split_documents(all_docs)

    # 加入向量库
    vector_store.add_documents(split_docs)
    vector_store.persist()
    return True, f"成功添加 {len(split_docs)} 个文本块，来自 {len(file_paths)} 个文件"

def rebuild_vector_store():
    """清空并重建向量库"""
    # 删除原有向量库
    if config.VECTOR_DB_DIR.exists():
        import shutil
        shutil.rmtree(config.VECTOR_DB_DIR)
    # 重新构建
    build_vector_store()
    return True, "向量库重建完成"