from fastapi import FastAPI
app = FastAPI(title="RAG Service")

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "rag-service", "version": "0.1.0"}
