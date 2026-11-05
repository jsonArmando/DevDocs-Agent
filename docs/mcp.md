# Reto A · Servidor MCP

El servidor ([`services/agent/app/mcp_server.py`](../services/agent/app/mcp_server.py)) expone:

| Herramienta | Qué hace | Costo LLM |
|---|---|---|
| `search_docs(query, k, doc_ids)` | Búsqueda híbrida + rerank en el índice | 0 |
| `ingest_document(path)` | Ingesta un archivo local (incremental) | 0 |
| `ask(question, thread_id)` | Pregunta al grafo agéntico completo | Igual que `/chat` |

## Claude Code
```bash
claude mcp add devdocs -- docker compose -f /ruta/a/devdocs-agent/docker-compose.yml exec -T agent python -m app.mcp_server
```
Prueba: *"Usa la herramienta search_docs de devdocs para buscar cómo declarar dependencias con yield"*.

## Claude Desktop (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "devdocs": {
      "command": "docker",
      "args": ["compose", "-f", "/ruta/a/devdocs-agent/docker-compose.yml", "exec", "-T", "agent", "python", "-m", "app.mcp_server"]
    }
  }
}
```
`ingest_document` recibe una ruta **dentro del contenedor**; para archivos de tu equipo, ejecuta el servidor sin Docker
(`cd services/agent && ../../.venv/bin/python -m app.mcp_server` con `INGESTION_URL=http://localhost:8001`).
