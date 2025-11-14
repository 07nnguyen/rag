import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder

COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
TOP_K = int(os.getenv("TOP_K", "6"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_qdrant: QdrantClient | None = None
SYSTEM_PROMPT = ("You are a helpful assistant. Answer ONLY from the provided context. "
"If the context is insufficient, say you don't have enough information. "
"Always include a short 'Nguồn' section listing file paths you used."
)
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def get_reranker():
    global _reranker
    if _reranker is None and RERANK_MODEL:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker

def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL)
    return _qdrant

def _filters(filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
    if not filters:
        return None
    return Filter(must=[FieldCondition(key=f"meta.{k}",
match=MatchValue(value=v)) for k, v in filters.items()])
    
def search(query: str, top_k: int, filters: Optional[dict]) -> List[Dict[str, Any]]:
    embs = get_embedder().encode([query], normalize_embeddings=True)
    res = get_qdrant().search(
        collection_name=COLLECTION,
        query_vector=embs[0].tolist(),
        limit=top_k * 3, # overfetch for rerank
        query_filter=_filters(filters),
        with_payload=True,
    )
    docs = []
    for p in res:
        payload = p.payload or {}
        docs.append({
            "text": payload.get("text", ""),
            "path": (payload.get("meta") or {}).get("path"),
            "score": float(p.score),
        })
        rr = get_reranker()
        if rr and docs:
            pairs = [(query, d["text"]) for d in docs]
            scores = rr.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            docs = [d for d, _ in ranked[:top_k]]
        else:
            docs = docs[:top_k]
        return docs
    
def build_messages(query: str, docs: List[Dict[str, Any]]):
    context = "\n\n---\n\n".join(d["text"][:1200] for d in docs)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
    ]

