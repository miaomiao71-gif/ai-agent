"""混合检索模块：BM25（关键词）+ 向量（语义）双路召回，RRF 融合，Rerank 重排序。

流程：
    query
      ├─ 向量检索（FAISS, top_k）──┐
      └─ BM25 检索（jieba, top_k）─┴─ RRF 倒数排名融合 ─ Rerank ─ top_n 结果

设计要点（面试讲解）：
1. 为什么混合：向量擅长语义相近但字面不同的 query，BM25 擅长专有名词 / 数字 / 精确词；
   二者互补，RRF 无需归一化分数即可融合（对各路分数分布无假设）。
2. 为什么 Rerank：双路召回是"粗排"，召回广但精度低；Cross-Encoder 类 Rerank 模型
   把 query 与 doc 拼在一起做交互编码，精度高，作为"精排"只处理少量候选，成本可控。
3. 降级策略：Rerank 依赖外部 API，失败时回退到 RRF 融合顺序，保证主链路可用。
"""

import json
import logging
import re

import jieba
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from config import settings

jieba.setLogLevel(logging.WARNING)

# 中文停用词（精简版，避免高频虚词稀释 BM25 权重）
STOP_WORDS = {
    "的", "了", "和", "是", "在", "我", "你", "他", "她", "它", "们", "这", "那",
    "有", "与", "及", "或", "为", "把", "被", "从", "对", "就", "都", "而", "也",
    "吗", "呢", "吧", "啊", "什", "么", "怎", "样", "哪", "个", "些", "之", "其",
    "以", "于", "上", "下", "中", "不", "无", "一", "个", "请", "问", "怎么", "什么",
}


def tokenize(text: str) -> list[str]:
    """中文分词 + 过滤停用词 / 单字 / 标点，供 BM25 建索引与查询。"""
    tokens = jieba.lcut(text.lower())
    return [
        t for t in tokens
        if len(t) > 1 and t not in STOP_WORDS and re.search(r"[\w\u4e00-\u9fff]", t)
    ]


class HybridRetriever:
    """混合检索器：向量 + BM25 + RRF + Rerank。"""

    def __init__(self) -> None:
        index_path = settings.index_path
        chunks_file = index_path / "chunks.json"
        if not index_path.exists() or not chunks_file.exists():
            raise FileNotFoundError(
                f"找不到索引 {index_path} 或 {chunks_file}。"
                "请先运行 python build_db.py 构建知识库。"
            )

        # Embedding 实例向量库与问答侧共用同一配置
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            base_url=settings.dashscope_base_url,
            api_key=settings.dashscope_api_key,
            check_embedding_ctx_length=False,
        )
        self.vectorstore = FAISS.load_local(
            str(index_path), self.embeddings, allow_dangerous_deserialization=True
        )

        # 从 chunks.json 还原 Document，并保证与向量库使用同一批 chunk_id
        with open(chunks_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        self.chunks: list[Document] = [
            Document(page_content=r["text"], metadata=r["metadata"]) for r in records
        ]
        self.chunk_by_id: dict[str, Document] = {
            d.metadata["chunk_id"]: d for d in self.chunks
        }
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.chunks])

    # ---------- 单路检索 ----------
    def vector_search(self, query: str, k: int) -> list[str]:
        """向量检索，返回按相似度排序的 chunk_id 列表。"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [d.metadata["chunk_id"] for d in docs]

    def bm25_search(self, query: str, k: int) -> list[str]:
        """BM25 关键词检索，返回按得分排序的 chunk_id 列表。"""
        scores = self.bm25.get_scores(tokenize(query))
        # argsort 升序取负号得到降序，再取前 k 个下标
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.chunks[i].metadata["chunk_id"] for i in top_idx]

    # ---------- 融合与精排 ----------
    @staticmethod
    def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
        """Reciprocal Rank Fusion：按名次累加 1/(k+rank)，与各路分数尺度无关。"""
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores, key=lambda cid: -scores[cid])

    def rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        """调用 DashScope Rerank 精排；失败时降级返回原顺序的前 top_n 条。"""
        if not docs:
            return []
        try:
            resp = requests.post(
                settings.rerank_api_url,
                headers={
                    "Authorization": f"Bearer {settings.dashscope_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.rerank_model,
                    "input": {
                        "query": query,
                        "documents": [d.page_content for d in docs],
                    },
                    "parameters": {"return_documents": False, "top_n": top_n},
                },
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json()["output"]["results"]
            ranked: list[Document] = []
            for item in results:
                doc = docs[item["index"]]
                doc.metadata["rerank_score"] = round(item["relevance_score"], 4)
                ranked.append(doc)
            return ranked
        except Exception as e:  # 网络 / 额度 / 格式异常统一降级
            print(f"[警告] Rerank 调用失败，已降级使用融合排序：{e}")
            return docs[:top_n]

    # ---------- 主入口 ----------
    def retrieve(self, query: str) -> list[Document]:
        """执行完整检索链路，返回最终用于生成答案的 Document 列表。"""
        top_k = settings.retrieve_top_k

        if settings.enable_hybrid:
            vec_ids = self.vector_search(query, top_k)
            bm25_ids = self.bm25_search(query, top_k)
            fused_ids = self.rrf_fuse([vec_ids, bm25_ids], k=settings.rrf_k)
        else:
            fused_ids = self.vector_search(query, top_k)

        candidates = [
            self.chunk_by_id[cid] for cid in fused_ids[:top_k] if cid in self.chunk_by_id
        ]

        if settings.enable_rerank:
            return self.rerank(query, candidates, settings.final_top_k)
        return candidates[: settings.final_top_k]
