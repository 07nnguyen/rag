import os, glob
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

def load_texts(data_dir: str) -> List[Dict]:
    items = []
    txt_like = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True) + glob.glob(os.path.join(data_dir, "**", "*.md"), recursive=True)
    pdf_like = glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)
    for p in txt_like:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                items.append({"text": f.read(), "meta": {"path":os.path.relpath(p, data_dir)}})
        except Exception as e:
            print(f"[WARN] can't read {p}: {e}")
            
    from pypdf import PdfReader
    for p in pdf_like:
        try:
            r = PdfReader(p)
            text = "\n".join([pg.extract_text() or "" for pg in r.pages])
            if text.strip():
                items.append({"text": text, "meta": {"path": os.path.relpath(p, data_dir)}})
        except Exception as e:
            print(f"[WARN] can't parse pdf {p}: {e}")
    return items

def chunk_items(items: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for it in items:
        for c in splitter.split_text(it["text"]):
            chunks.append({"text": c, "meta": it["meta"]})
    return chunks

def ensure_collection(client: QdrantClient, dim: int):
    if not client.collection_exists(COLLECTION):
        client.recreate_collection(collection_name=COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE),)

def main():
    data_dir = "/data"
    print(f"[INGEST] loading from {data_dir} …")
    items = load_texts(data_dir)
    if not items:
        print("[INGEST] no files found. Put .txt/.md/.pdf into ./data and rerun.")
        return
    
    print(f"[INGEST] {len(items)} docs loaded. chunking …")
    chunks = chunk_items(items)
    texts = [c["text"] for c in chunks]
    print(f"[INGEST] total chunks: {len(texts)}")
    print(f"[EMBED] model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=64,
    show_progress_bar=True)
    dim = len(vecs[0])
    print(f"[QDRANT] connect {QDRANT_URL}, dim={dim}")
    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client, dim)
    points = []
    for i, (v, c) in enumerate(zip(vecs, chunks)):
        payload = {"text": c["text"], "meta": c["meta"]}
        points.append(PointStruct(id=i, vector=v.tolist(), payload=payload))
    print(f"[UPSERT] {len(points)} points …") 
    client.upsert(collection_name=COLLECTION, points=points)
    print("[DONE] ingestion finished.")

if __name__ == "__main__":
    main()
