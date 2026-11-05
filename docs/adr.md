# Architecture Decision Records

## ADR-001 · LangGraph como framework de orquestación
**Contexto**: el enunciado exige grafo propio con estado tipado, arista condicional, ciclo controlado, memoria por hilo y delegación.
**Decisión**: LangGraph `StateGraph` con `TypedDict`, subgrafos por agente, checkpointer y `stream(subgraphs=True)`.
**Alternativas**: LlamaIndex Workflows (eventos, menos soporte de memoria persistente y exportación de diagramas); PydanticAI Graph (más joven); agentes prefabricados (prohibidos por las reglas).
**Consecuencias**: (+) memoria, streaming y diagrama gratis; (−) acoplamiento a la API de LangGraph, mitigado con nodos puros sobre dicts y puertos (`AgentLLM`, `RetrieverPort`) que se prueban sin el framework.

## ADR-002 · Qdrant como base vectorial
**Decisión**: Qdrant con vector denso (HNSW m=16, ef_construct=128, hnsw_ef=96), vector disperso BM25 con IDF, fusión RRF en el servidor y payload indexes en `doc_id`, `source`, `content_hash`.
**Alternativas**: pgvector (sin BM25 nativo, habría que combinar con `tsvector`); Chroma (sin híbrido nativo); Milvus (más pesado de operar); Pinecone (pago y no local).
**Consecuencias**: búsqueda híbrida en una sola consulta; filtros por documento eficientes; un contenedor liviano.

## ADR-003 · Chunking estructural con tamaño acotado
**Decisión**: secciones por encabezados (MD) o página (PDF), ~1 400 caracteres, solape ~200 al partir, bloques de código indivisibles, prefijo de contexto en el texto vectorizado.
**Alternativas**: tamaño fijo (corta código y pierde la sección), semántico por embeddings (más costo en ingesta, sin ganancia clara en documentación ya estructurada).
**Consecuencias**: secciones citables precisas; recall mejorado por el prefijo; ajustable por `.env`.

## ADR-004 · Niveles de modelo y proveedor intercambiable
**Decisión**: modelo rápido (Claude Haiku) para router, grader en lote, rewriter, planner y juez; modelo grande (Claude Sonnet) para generar y sintetizar. `init_chat_model` permite Anthropic, OpenAI, Groq u Ollama sin cambiar código.
**Alternativas**: un solo modelo grande (costo alto en decisiones simples); un solo modelo pequeño (peor redacción y citas).
**Consecuencias**: costo por consulta dominado por una sola llamada grande; modo local con Ollama a costo de API cero.

## ADR-005 · Embeddings, BM25 y reranker locales con fastembed
**Decisión**: `BAAI/bge-small-en-v1.5`, `Qdrant/bm25` y `Xenova/ms-marco-MiniLM-L-6-v2` en ONNX/CPU, cacheados en un volumen.
**Alternativas**: embeddings por API (costo y latencia de red); sentence-transformers con PyTorch (imagen ~2 GB más pesada).
**Consecuencias**: costo cero, imágenes livianas; primer arranque descarga los modelos (~150 MB).

## ADR-006 · Memoria con checkpointer SQLite
**Decisión**: `SqliteSaver` en un volumen; `messages` con `add_messages`; el supervisor reescribe la pregunta autónoma con el historial reciente.
**Alternativas**: Postgres (`PostgresSaver`) para múltiples réplicas: cambio de una línea en `container.py` cuando se escale horizontalmente.

## ADR-007 · Investigación con DuckDuckGo por defecto
**Decisión**: proveedor por Strategy: DuckDuckGo (gratis) o Tavily (con key). Timeout duro, clasificación de errores y reintento con la consulta literal.
**Consecuencias**: demo sin costo; en producción conviene Tavily por estabilidad y límites de uso.

## ADR-008 · Evaluación con juez propio y métricas deterministas
**Decisión**: una llamada de juez por respuesta (faithfulness, relevancy, correctness, relevancia de cada contexto) + hit@5/MRR por fuente esperada para la ablación de recuperación.
**Alternativas**: Ragas/DeepEval (más llamadas por métrica, más dependencias).
**Consecuencias**: evaluación barata y reproducible; el juez debería calibrarse con una muestra revisada a mano.

## ADR-009 · Fuera de alcance consciente
Human-in-the-loop (reto D) y autenticación de la API no se implementan en esta entrega; el diseño los admite (`interrupt()` de LangGraph antes de `research_agent`; gateway con OAuth delante de los servicios).
