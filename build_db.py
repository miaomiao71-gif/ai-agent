"""离线索引构建：加载 → 切分 → 增强元数据 → 向量化 → 持久化。

产物（均在 faiss_index/ 下）：
- index.faiss / index.pkl : FAISS 向量索引（供向量检索）
- chunks.json            : 全部 chunk 的正文 + 元数据（供 BM25 检索与段落级溯源）

面试讲解要点：
- 向量检索与 BM25 检索共用同一批 chunk，用 chunk_id 一一对应，融合时才能对齐；
- 切分阶段即写入来源元数据（文件名 / 页码 / 序号），溯源信息不依赖运行时猜测。
"""

import json
import sys
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def load_documents() -> list:
    """按后缀加载知识库目录下的全部文档。"""
    kb_path = settings.kb_path
    print(f"开始扫描知识库目录: {kb_path} ...")
    loaders = [
        DirectoryLoader(str(kb_path), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=False),
        DirectoryLoader(
            str(kb_path), glob="**/*.txt", loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}, show_progress=False,
        ),
        DirectoryLoader(
            str(kb_path), glob="**/*.md", loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}, show_progress=False,
        ),
        DirectoryLoader(str(kb_path), glob="**/*.docx", loader_cls=Docx2txtLoader, show_progress=False),
    ]

    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:  # 单个加载器失败不影响其余文档
            print(f"[警告] 加载部分文档时出错（已跳过）：{e}")

    if not docs:
        sys.exit(f"错误：在 {kb_path} 目录下没有找到支持的文档。")
    return docs


def enrich_metadata(chunks: list) -> None:
    """为每个 chunk 补充溯源元数据（原地修改）。

    字段：
    - chunk_id   : 全局唯一序号，如 0001（BM25 / 向量结果据此对齐）
    - source     : 原始文件绝对路径
    - file_name  : 文件名（溯源展示用）
    - file_type  : 扩展名
    - page       : PDF 页码（其他格式记为 -）
    - chunk_index: 该文档内的第几块（段落级定位）
    - char_start / char_end : 在原文字符流中的大致位置
    """
    doc_chunk_counter: dict[str, int] = {}
    char_cursor = 0
    for idx, chunk in enumerate(chunks):
        src = chunk.metadata.get("source", "unknown")
        path = Path(src)
        doc_index = doc_chunk_counter.get(src, 0)
        doc_chunk_counter[src] = doc_index + 1

        char_start = char_cursor
        char_cursor += len(chunk.page_content)
        chunk.metadata.update(
            {
                "chunk_id": f"{idx:04d}",
                "source": src,
                "file_name": path.name or "未知文件",
                "file_type": path.suffix.lower().lstrip(".") or "未知",
                "page": chunk.metadata.get("page", -1),
                "chunk_index": doc_index,
                "char_start": char_start,
                "char_end": char_cursor,
            }
        )


def build_database() -> None:
    docs = load_documents()
    print(f"成功加载 {len(docs)} 页/份文档，正在切分...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    enrich_metadata(chunks)
    print(f"切分为 {len(chunks)} 个文本块，已补充溯源元数据。")

    # 向量化并保存 FAISS
    print("正在调用 Embedding 模型进行向量化（请稍候）...")
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.dashscope_base_url,
        api_key=settings.dashscope_api_key,
        check_embedding_ctx_length=False,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_path: Path = settings.index_path
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))

    # 落盘 chunk 正文 + 元数据，供 BM25 使用（保证与向量库同批数据）
    chunks_file = index_path / "chunks.json"
    records = [
        {"chunk_id": c.metadata["chunk_id"], "text": c.page_content, "metadata": c.metadata}
        for c in chunks
    ]
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ 向量库构建完成！索引与 {chunks_file.name} 已保存至 {index_path} 目录。")


if __name__ == "__main__":
    build_database()
