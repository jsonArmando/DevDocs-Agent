# Guion del video demo (≤ 5 min)

| Tiempo | Escena | Qué mostrar | Consulta |
|---|---|---|---|
| 0:00–0:30 | Arquitectura | Diagramas del README, `docker compose ps` con 4 servicios sanos | — |
| 0:30–1:10 | Ingesta | Subir un PDF y dos MD en el chat; volver a subir uno → `duplicate` (ingesta incremental) | — |
| 1:10–2:00 | Respuesta con citas | Agente 📚 Documentación, pasos `supervisor → retrieve → grade → generate`, fuentes con sección y página | *How do I enable CORS in FastAPI?* y seguimiento *and with credentials?* (memoria) |
| 2:00–2:50 | Rewriter | Pasos con `rewrite → retrieve`, consulta reescrita visible | *¿cómo hago que mi endpoint devuelva un 404 cuando algo no existe?* |
| 2:50–3:50 | Delegación a investigación | `doc_agent.no_context → research_agent.search → synthesize`, prefijo 🌐 y enlaces | *What is the latest stable release of Kubernetes?* |
| 3:50–4:20 | Fuera de alcance | Respuesta directa, 0 tokens en métricas | *Recomiéndame una receta de ajiaco* |
| 4:20–5:00 | Evaluación y MCP | `eval/results/report.md` (lineal vs grafo, ablación) y una llamada `search_docs` desde Claude Code | — |

Consejo: antes de grabar, ejecuta las consultas una vez para que los modelos locales ya estén descargados.
