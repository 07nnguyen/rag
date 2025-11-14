# app/main.py
# -*- coding: utf-8 -*-
import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
TOP_K = int(os.getenv("TOP_K", "6"))

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "mixedbread-ai/mxbai-embed-large-v1")

# === clients ===
_qdrant: QdrantClient | None = None
_embedder: SentenceTransformer | None = None


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL)
    return _qdrant


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def embed(text: str) -> List[float]:
    m = get_embedder()
    vec = m.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


class QueryReq(BaseModel):
    query: str
    top_k: int | None = None


class Source(BaseModel):
    text: str
    score: float
    path: str | None = None


class QueryResp(BaseModel):
    answer: str
    sources: List[Source]


app = FastAPI(title="RAG Service", version="0.3.0")


@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "rag-service", "version": "0.3.0"}


@app.post("/rag/query", response_model=QueryResp)
def rag_query(req: QueryReq):
    top_k = req.top_k or TOP_K

    qvec = embed(req.query)
    client = get_qdrant()
    result: List[ScoredPoint] = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qvec,
        limit=top_k,
    )

    if not result:
        return QueryResp(
            answer="Không tìm thấy ngữ cảnh phù hợp trong kho dữ liệu.",
            sources=[],
        )

    contexts = []
    sources: List[Source] = []
    for p in result:
        payload = p.payload or {}
        text = payload.get("text", "")
        meta = payload.get("meta", {}) or {}
        path = meta.get("path")
        contexts.append(text)
        sources.append(Source(text=text[:2000], score=float(p.score), path=path))

    context_str = "\n\n---\n\n".join(contexts)

    # Ở đây bạn có thể gọi Ollama để generate câu trả lời.
    # Để đơn giản tạm thời: trả luôn context + câu hỏi.
    answer = f"Câu hỏi: {req.query}\n\nDưới đây là các đoạn phù hợp:\n\n{context_str}"

    return QueryResp(answer=answer, sources=sources)
