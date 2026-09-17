#!/usr/bin/env bash
# Start Postgres 17 + pgvector in Docker on port 5433, create the venv, ingest, evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."
docker rm -f rag-pg >/dev/null 2>&1 || true
docker run -d --name rag-pg -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag -p 5433:5432 pgvector/pgvector:pg17 >/dev/null
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
sleep 3
.venv/bin/python -m rag.ingest
.venv/bin/python -m rag.evaluate
