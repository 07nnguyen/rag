import os, glob, json
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ====== cấu hình cơ bản ======
COLLECTION = "docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"  # nhanh + chất lượng tốt

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# ====== helpers ======
def load_texts(data_dir: str) -> List[Dict]:
    """Đọc file .txt, .md, .pdf → trả list dict {text, meta}"""
    items = []
    txt_like = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True) + \
               glob.glob(os.path.join(data_dir, "**", "*.md"), recursive=True)
    pdf_like = glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)

    for p in txt_like:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                items.append({"text": f.read(), "meta": {"path": p}})
        except Exception as e:
            print(f"[WARN] can't read {p}: {e}")

    # PDF đọc nhanh gọn bằng pypdf
    from pypdf import PdfReader
    for p in pdf_like:
        try:
            r = PdfReader(p)
            text = "\n".join([pg.extract_text() or "" for pg in r.pages])
            if text.strip():
                items.append({"text": text, "meta": {"path": p}})
        except Exception as e:
            print(f"[WARN] can't parse pdf {p}: {e}")
    return items

def chunk_items(items: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for it in items:
        for c in splitter.split_text(it["text"]):
            chunks.append({"text": c, "meta": it["meta"]})
    return chunks

def ensure_collection(client: QdrantClient, dim: int):
    if not client.collection_exists(COLLECTION):
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

def main():
    data_dir = "/data"
    print(f"[INGEST] loading from {data_dir} ...")
    items = load_texts(data_dir)
    if not items:
        print("[INGEST] no files found. Put .txt/.md/.pdf into ./data and re-run.")
        return

    print(f"[INGEST] {len(items)} docs loaded. chunking ...")
    chunks = chunk_items(items)
    texts = [c["text"] for c in chunks]
    print(f"[INGEST] total chunks: {len(texts)}")

    print(f"[EMBED] model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    dim = len(vecs[0])

    print(f"[QDRANT] connect {QDRANT_HOST}:{QDRANT_PORT}, dim={dim}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client, dim)

    points = []
    for i, (v, c) in enumerate(zip(vecs, chunks)):
        payload = {
            "text": c["text"],
            "meta": c["meta"],
        }
        points.append(PointStruct(id=i, vector=v.tolist(), payload=payload))

    print(f"[UPSERT] {len(points)} points ...")
    client.upsert(collection_name=COLLECTION, points=points)
    print("[DONE] ingestion finished.")

if __name__ == "__main__":
    main()
