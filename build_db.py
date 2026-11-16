import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

KB_PATH = "./knowledge_base"
INDEX_PATH = "./faiss_index"
EMBEDDING_MODEL = "qwen3.7-text-embedding-flash"


def build_database():
    print(f"开始扫描知识库目录: {KB_PATH} ...")

    # 1. 加载文档
    loaders = [
        DirectoryLoader(KB_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=False),
        DirectoryLoader(KB_PATH, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}, show_progress=False),
        DirectoryLoader(KB_PATH, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}, show_progress=False),
        DirectoryLoader(KB_PATH, glob="**/*.docx", loader_cls=Docx2txtLoader, show_progress=False),
    ]

    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"[警告] 加载部分文档时出错（已跳过）：{e}")

    if not docs:
        sys.exit(f"错误：在 {KB_PATH} 目录下没有找到支持的文档。")

    # 2. 文本切分
    print(f"成功加载 {len(docs)} 页/份文档，正在切分...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"切分为 {len(chunks)} 个文本块。")

    # 3. 向量化并保存到本地
    print("正在调用 Embedding 模型进行向量化（请稍候）...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        check_embedding_ctx_length=False
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 保存到本地文件夹
    vectorstore.save_local(INDEX_PATH)
    print(f"✅ 向量库构建完成！已保存至 {INDEX_PATH} 目录。")


if __name__ == "__main__":
    build_database()