import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path

COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
TOP_K = int(os.getenv("TOP_K", "6"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_qdrant: QdrantClient | None = None

SYSTEM_PROMPT = Path("app/service/prompts/system_hr.txt").read_text(encoding="utf-8")

# -------------------------
# Helpers
# -------------------------

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
    return Filter(
        must=[
            FieldCondition(
                key=f"meta.{k}",
                match=MatchValue(value=v)
            )
            for k, v in filters.items()
        ]
    )

# -------------------------
# IDP-aware retrieval
# -------------------------

def is_idp_query(q: str) -> bool:
    ql = q.lower()
    keywords = [
        "idp", "kế hoạch phát triển", "lộ trình",
        "3 năm", "phát triển năng lực",
        "đào tạo", "career", "hrbp"
    ]
    return any(k in ql for k in keywords)

def compute_k(query: str, user_top_k: int) -> tuple[int, int]:
    if is_idp_query(query):
        search_k = max(user_top_k * 2, 12)
        final_k  = max(user_top_k, 8)
    else:
        search_k = max(user_top_k * 2, 6)
        final_k  = user_top_k
    return search_k, final_k

# -------------------------
# Vector search + rerank
# -------------------------

def search(query: str, top_k: int, filters: Optional[dict]) -> List[Dict[str, Any]]:
    search_k, final_k = compute_k(query, top_k)

    embs = get_embedder().encode([query], normalize_embeddings=True)
    hits = get_qdrant().search(
        collection_name=COLLECTION,
        query_vector=embs[0].tolist(),
        limit=search_k,
        query_filter=_filters(filters),
        with_payload=True,
    )

    # collect all docs first
    docs = []
    for h in hits:
        payload = h.payload or {}
        meta = payload.get("meta") or {}
        docs.append({
            "text": payload.get("text", ""),
            "meta": meta,
            "score": float(h.score),
        })

    # rerank
    rr = get_reranker()
    if rr and docs:
        pairs = [(query, d["text"]) for d in docs]
        scores = rr.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        docs = [d for d, _ in ranked[:final_k]]
    else:
        docs = docs[:final_k]

    return docs

# -------------------------
# Prompt building
# -------------------------

def build_messages(query: str, docs: List[Dict[str, Any]]):
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.get("meta") or {}
        path = meta.get("path", "UNKNOWN_SOURCE")
        score = d.get("score")
        text = (d.get("text") or "").strip()

        header = [f"[DOC {i}]", f"SOURCE: {path}"]
        if isinstance(score, (int, float)):
            header.append(f"SCORE: {score:.4f}")

        blocks.append(
            "\n".join(header) + "\nTEXT:\n" + text[:1200]
        )

    context = "\n\n---\n\n".join(blocks)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
    ]
