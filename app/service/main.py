import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import io
from pypdf import PdfReader

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

def extract_pdf_text(file: UploadFile, max_chars: int = 6000) -> str:
    # đọc bytes từ UploadFile
    data = file.file.read()
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for pg in reader.pages:
        t = pg.extract_text() or ""
        if t.strip():
            parts.append(t)
        if sum(len(x) for x in parts) >= max_chars:
            break
    text = "\n".join(parts)
    return text[:max_chars]

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
    
@app.post("/rag/query_with_pdf", response_model=QueryResp)
def rag_query_with_pdf(
    query: str = Form(..., min_length=2, max_length=4000),
    top_k: int = Form(TOP_K),
    file: UploadFile = File(...),
    # filters: Optional[str] = Form(None)  # nếu muốn sau này nhận filters dạng JSON string
):
    # 0) extract user pdf text
    if file.content_type not in ("application/pdf",):
        raise HTTPException(status_code=400, detail="Only PDF is supported")

    user_pdf_text = extract_pdf_text(file, max_chars=6000)
    if not user_pdf_text.strip():
        raise HTTPException(status_code=400, detail="PDF has no extractable text (maybe scanned image PDF)")

    # 1) retrieve from Qdrant như cũ
    vq = embedder().encode([query], normalize_embeddings=True)[0].tolist()
    res = qdrant().search(
        collection_name=COLLECTION,
        query_vector=vq,
        limit=max(1, int(top_k)),
        query_filter=None,
        with_payload=True
    )

    docs = []
    for p in res or []:
        payload = p.payload or {}
        docs.append(Source(
            text=(payload.get("text") or "")[:300],
            path=(payload.get("meta") or {}).get("path"),
            score=float(p.score)
        ))

    # 2) build combined context: USER PDF + RAG CONTEXT
    rag_context = "\n\n---\n\n".join([(p.payload or {}).get("text","") for p in (res or [])])[:4000]

    combined_context = (
        "[USER PROVIDED ASSESSMENT PDF]\n"
        f"{user_pdf_text}\n\n"
        "---\n\n"
        "[REFERENCE DOCUMENTS]\n"
        f"{rag_context}"
    )

    messages = [
        {"role": "system", "content":
            "You are an HR assistant. Treat [USER PROVIDED ASSESSMENT PDF] as FACT input. "
            "Answer ONLY using the provided context blocks. If insufficient, say THIẾU DỮ LIỆU and list what's missing. "
            "Do NOT invent courses/programs. Provide citations by referring to the SOURCE filenames from reference docs when possible."
        },
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{combined_context}"}
    ]

    resp = llm().chat.completions.create(model=GEN_MODEL, messages=messages, temperature=0.2)
    answer = resp.choices[0].message.content or ""

    return QueryResp(answer=answer, sources=docs)
