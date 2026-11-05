.PHONY: help up up-local down logs dataset ingest eval eval-quick test lint graph dev-install dev-ingestion dev-agent dev-frontend

help:            ## Lista de comandos
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

up:              ## Levanta todo con Docker (LLM en la nube según .env)
	docker compose up -d --build

up-local:        ## Levanta todo 100 % local con Ollama y descarga el modelo
	docker compose --profile local up -d --build
	docker compose exec ollama ollama pull qwen2.5:7b

down:            ## Detiene los servicios
	docker compose down

logs:            ## Logs de agente e ingesta
	docker compose logs -f agent ingestion

dataset:         ## Descarga el dataset público a data/raw
	python scripts/fetch_dataset.py

ingest:          ## Ingesta data/raw (incremental: re-ejecutar no duplica)
	python scripts/ingest_dataset.py

eval:            ## Evaluación completa: RAG lineal vs grafo + ablación de recuperación
	docker compose --profile eval run --rm eval

eval-quick:      ## Evaluación sin juez LLM (enrutamiento, latencia, costo, ablación)
	docker compose --profile eval run --rm eval python eval/run_eval.py --skip-judge

graph:           ## Exporta el diagrama Mermaid del grafo
	curl -s localhost:8002/graph > docs/graph.mmd && echo "docs/graph.mmd"

test:            ## Tests unitarios (sin servicios externos)
	python -m pytest -q tests

lint:
	ruff check .

# ---------------------------------------------------------------- ejecución local SIN Docker
dev-install:     ## Crea .venv con todas las dependencias
	python -m venv .venv && .venv/bin/pip install -e libs/core -r services/ingestion/requirements.txt \
	  -r services/agent/requirements.txt -r frontend/requirements.txt pytest ruff

dev-ingestion:   ## Ingesta en localhost:8001 (requiere Qdrant: docker run -p 6333:6333 qdrant/qdrant)
	cd services/ingestion && QDRANT_URL=http://localhost:6333 MODELS_CACHE_DIR=../../.models ../../.venv/bin/uvicorn app.main:app --port 8001 --reload

dev-agent:       ## Agente en localhost:8002
	cd services/agent && QDRANT_URL=http://localhost:6333 MODELS_CACHE_DIR=../../.models CHECKPOINT_DB=../../.data/checkpoints.sqlite \
	  OLLAMA_BASE_URL=http://localhost:11434 INGESTION_URL=http://localhost:8001 AGENT_URL=http://localhost:8002 \
	  ../../.venv/bin/uvicorn app.main:app --port 8002 --reload

dev-frontend:    ## Chat en localhost:8501
	.venv/bin/streamlit run frontend/app.py
