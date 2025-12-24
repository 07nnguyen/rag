import os, glob, hashlib
from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

DATA_DIR = os.getenv("DATA_DIR", "/data")


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def make_doc_id(rel_path: str) -> str:
    # ổn định theo đường dẫn tương đối trong /data
    return sha1_hex(rel_path)[:16]


def make_chunk_id(doc_id: str, chunk_index: int, chunk_text: str) -> str:
    # ổn định theo doc + index + hash nội dung (để chống thay đổi nhỏ)
    h = sha1_hex(chunk_text)[:16]
    return f"{doc_id}:{chunk_index}:{h}"


def load_texts(data_dir: str) -> List[Dict]:
    items = []
    txt_like = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True) + \
               glob.glob(os.path.join(data_dir, "**", "*.md"), recursive=True)
    pdf_like = glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)

    for p in txt_like:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                rel = os.path.relpath(p, data_dir)
                items.append({"text": f.read(), "meta": {"path": rel, "type": os.path.splitext(rel)[1].lstrip(".")}})
        except Exception as e:
            print(f"[WARN] can't read {p}: {e}")

    from pypdf import PdfReader
    for p in pdf_like:
        try:
            r = PdfReader(p)
            text = "\n".join([(pg.extract_text() or "") for pg in r.pages])
            if text.strip():
                rel = os.path.relpath(p, data_dir)
                items.append({"text": text, "meta": {"path": rel, "type": "pdf"}})
        except Exception as e:
            print(f"[WARN] can't parse pdf {p}: {e}")

    return items


def chunk_items(items: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for it in items:
        rel_path = it["meta"]["path"]
        doc_id = make_doc_id(rel_path)

        split = splitter.split_text(it["text"])
        for idx, c in enumerate(split):
            meta = dict(it["meta"])
            meta.update({
                "doc_id": doc_id,
                "chunk_index": idx,
            })
            chunks.append({"text": c, "meta": meta})
    return chunks


def ensure_collection(client: QdrantClient, dim: int):
    # chỉ tạo nếu chưa có; không reset để giữ dữ liệu cũ
    if not client.collection_exists(COLLECTION):
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def main():
    print(f"[INGEST] loading from {DATA_DIR} …")
    items = load_texts(DATA_DIR)
    if not items:
        print("[INGEST] no files found. Put .txt/.md/.pdf into ./data and rerun.")
        return

    print(f"[INGEST] {len(items)} docs loaded. chunking …")
    chunks = chunk_items(items)
    texts = [c["text"] for c in chunks]
    print(f"[INGEST] total chunks: {len(texts)}")

    print(f"[EMBED] model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )

    dim = len(vecs[0])
    print(f"[QDRANT] connect {QDRANT_URL}, dim={dim}")
    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client, dim)

    points = []
    for v, c in zip(vecs, chunks):
        doc_id = c["meta"]["doc_id"]
        chunk_index = int(c["meta"]["chunk_index"])
        pid = make_chunk_id(doc_id, chunk_index, c["text"])

        payload = {
            "text": c["text"],
            "meta": c["meta"],
        }
        points.append(PointStruct(id=pid, vector=v.tolist(), payload=payload))

    print(f"[UPSERT] {len(points)} points …")
    client.upsert(collection_name=COLLECTION, points=points)

    print("[DONE] ingestion finished.")


if __name__ == "__main__":
    main()
