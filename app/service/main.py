import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ===== env =====
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION  = os.getenv("QDRANT_COLLECTION", "docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
TOP_K       = int(os.getenv("TOP_K", "6"))

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "dummy")
GEN_MODEL       = os.getenv("GEN_MODEL", "llama3.2")

# ===== lazy singletons =====
_qdrant = None
_embedder = None
_client = None

def qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL, timeout=60.0)
    return _qdrant

def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def llm():
    global _client
    if _client is None:
        kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        _client = OpenAI(**kwargs)
    return _client

# ===== models =====
class QueryReq(BaseModel):
    query: str = Field(..., min_length=2, max_length=4000)
    top_k: int = TOP_K
    filters: Optional[Dict[str, Any]] = None

class Source(BaseModel):
    text: str
    path: Optional[str] = None
    score: float

class QueryResp(BaseModel):
    answer: str
    sources: List[Source]

# ===== app =====
app = FastAPI(title="RAG Service")

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "rag-service", "version": "0.3.0"}

def _filters(flt: Optional[Dict[str, Any]]):
    if not flt: return None
    return Filter(must=[FieldCondition(key=f"meta.{k}", match=MatchValue(value=v)) for k, v in flt.items()])

@app.post("/rag/query", response_model=QueryResp)
def rag_query(req: QueryReq):
    # 1) embed query + search
    vq = embedder().encode([req.query], normalize_embeddings=True)[0].tolist()
    res = qdrant().search(collection_name=COLLECTION, query_vector=vq, limit=max(1, req.top_k), query_filter=_filters(req.filters), with_payload=True)
    if not res:
        return QueryResp(answer="Không tìm th?y ng? c?nh phù h?p trong kho d? li?u.", sources=[])

    docs = []
    for p in res:
        payload = p.payload or {}
        docs.append(Source(text=(payload.get("text") or "")[:300], path=(payload.get("meta") or {}).get("path"), score=float(p.score)))

    # 2) build prompt (ng?n g?n) và g?i LLM
    context = "\n\n---\n\n".join([(p.payload or {}).get("text","") if hasattr(p,"payload") else "" for p in res])[:4000]
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer ONLY from the provided context. If insufficient, say you don't know."},
        {"role": "user", "content": f"Question: {req.query}\n\nContext:\n{context}"}
    ]
    resp = llm().chat.completions.create(model=GEN_MODEL, messages=messages, temperature=0.2)
    answer = resp.choices[0].message.content or ""

    return QueryResp(answer=answer, sources=docs)
