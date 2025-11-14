# RAG Stack (per diagram)


## Prereqs
- Docker Engine + Compose
- GHCR access (login with PAT to pull private images)


## 1) Configure
Copy `.env.example` → `.env` and fill keys (OPENAI_API_KEY, etc.).


## 2) Run (production compose)
```bash
docker compose --env-file .env pull
docker compose --env-file .env up -d qdrant redis litellm postgres langfuse
# ingest once when /data has files
docker compose --env-file .env run --rm ingest
# bring up the API + UI
docker compose --env-file .env up -d rag-service open-webui