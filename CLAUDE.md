# MCP KQL Server

AI-powered MCP server for executing KQL queries against Azure Data Explorer with NL2KQL conversion, schema memory, and a self-improving learning loop.

## Quick Reference

- **Language:** Python 3.10+
- **Package manager:** uv (use `uv run --python 3.12` — Python 3.14 is NOT compatible with pydantic-core)
- **Framework:** FastMCP
- **Entry point:** `mcp_kql_server/mcp_server.py:main()`
- **Transport:** STDIO (default), SSE, or streamable-HTTP

## Project Structure

```
mcp_kql_server/
  mcp_server.py        # FastMCP server, 3 MCP tools, NL2KQL pipeline
  execute_kql.py       # Kusto query execution, retry logic, background learning loop
  memory.py            # PostgreSQL + pgvector memory (schemas, queries, join hints, cache)
  utils.py             # SchemaManager (3-strategy discovery), ErrorHandler, retry
  kql_validator.py     # Pre-execution validation (tables, columns, operator syntax)
  ai_prompts.py        # System prompts, few-shot examples, prompt builders
  llm_client.py        # Azure OpenAI client (NL2KQL generation + embeddings)
  observability.py     # Langfuse tracing (wraps OpenAI calls, MCP tools, Kusto execution)
  constants.py         # Config, KQL reserved words, operator rules, env var mapping
  performance.py       # Connection pooling (singleton), batch execution, health checks
  rest_api.py          # Starlette REST endpoints (/query, /health, /tables, /schema, etc.)
  kql_auth.py          # Azure CLI auth with device-code fallback
  version_checker.py   # PyPI version check at startup
```

## MCP Tools (the public API)

1. **`execute_kql_query`** — Main tool. Accepts raw KQL or natural language (`generate_query=True`). Validates, executes, caches results, learns in background.
2. **`list_tables`** — Lists all tables in a database.
3. **`schema_memory`** — Schema management: discover, refresh, get_context, get_stats, enrich_descriptions, generate_report.

## Running Tests

```bash
uv run --python 3.12 pytest tests/ -q --ignore=tests/test_dynamic_fields.py --ignore=tests/test_efficiency.py
```

The two ignored tests have pre-existing failures unrelated to current work. All other tests (56+) should pass.

## Dependencies

Defined in three places — keep them in sync:
- `pyproject.toml` (source of truth)
- `requirements.txt` (for `pip install -r`)
- `deployment/requirements-prod.txt` (slim, no azure-cli — uses service principal auth)

## Key Environment Variables

### Required
| Variable | Purpose |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI API endpoint |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | API version (e.g., "2024-02-01") |
| `AZURE_OPENAI_DEPLOYMENT` | Chat model deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment name |
| `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_DATABASE` | PostgreSQL connection |

### Observability (Langfuse)
| Variable | Purpose |
|---|---|
| `LANGFUSE_BASEURL` | Langfuse host (e.g., `https://cloud.langfuse.com`) |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_TRACING_ENVIRONMENT` | Environment tag for filtering traces |

Langfuse is a hard dependency. All LLM calls (NL2KQL, embeddings, descriptions) are automatically traced via the `langfuse.openai` drop-in wrapper. MCP tools and Kusto execution are traced via `@observe()` decorators.

### Optional
| Variable | Purpose |
|---|---|
| `MCP_TRANSPORT` | `stdio` (default), `sse`, or `streamable-http` |
| `MCP_HOST` / `MCP_PORT` | Host/port for SSE/HTTP transport |
| `KUSTO_CLUSTER_URL` / `KUSTO_DEFAULT_DATABASE` | Defaults for REST API |
| `KUSTO_CLIENT_ID` / `KUSTO_CLIENT_SECRET` / `KUSTO_TENANT_ID` | Service principal auth (prod) |

## Deployment

Deployed as a sidecar container in the nexus Helm chart:
- **Chart:** `shift-platform/.../nexus-chart/templates/kql-mcp-server-deployment.yaml`
- **Image:** Dockerfile in `deployment/`
- **Secrets:** Shared via the nexus externalSecrets target (Langfuse keys, DB creds, Azure OpenAI keys)

## NL2KQL Pipeline (`mcp_server.py:_generate_kql_from_natural_language`)

When `execute_kql_query` is called with `generate_query=True`, the pipeline runs:

1. **Semantic table discovery** — `memory_manager.find_relevant_tables()` uses pgvector cosine similarity to find up to 15 candidate tables
2. **Schema resolution** — If `table_name` is explicit, use it directly. Otherwise use semantic results, or fall back to `_get_database_schema()` (5-min cache)
3. **Few-shot context** — `find_similar_queries()` retrieves up to 3 past successful queries; `get_relevant_context()` gets CAG context
4. **LLM generation** — Two paths:
   - *Single table*: `build_generation_prompt()` + `KQL_SYSTEM_PROMPT` → `generate_kql()`
   - *Multiple candidates*: `build_multi_table_prompt()` + `MULTI_TABLE_SYSTEM_PROMPT` → LLM picks table AND writes query
5. **Schema validation** — All columns in generated query are checked against the discovered schema. KQL reserved words are filtered out
6. **Fallback (LLM unavailable)** — Heuristic generation: matches NL words against column names, handles dynamic fields with type accessors, produces `TableName | project matched_cols | take 10`

Generation method is returned as: `llm_generated`, `llm_multi_table`, `schema_memory_validated`, or `schema_memory_fallback`.

## Query Execution & Learning Loop (`execute_kql.py`)

**Execution flow:**
1. Auth check → cache check (120s TTL) → `kql_validator.validate_query()` → `_execute_kusto_query_sync()`
2. On SEM0100 errors (missing column): auto-triggers schema refresh for involved tables
3. Results serialized to JSON with special type handling (timestamps → ISO, NaN → null)

**Background learning** (async, non-blocking — fires after every successful query):
1. `extract_tables_from_query()` → parse table names from KQL
2. `add_successful_query()` → store query + embedding in PostgreSQL
3. `_ensure_schema_discovered()` → discover/refresh schemas for any new tables
4. `_discover_column_name_joins()` → learn join hints from shared column names
5. `_learn_dynamic_field_patterns()` → learn JSON sub-field access patterns

## Memory System (`memory.py`)

PostgreSQL + pgvector with 5 tables (prefixed `kql_mcp_`):
- `schemas` — table metadata + 1536-dim embeddings (HNSW index)
- `queries` — successful query history + embeddings for few-shot retrieval
- `join_hints` — discovered join relationships between tables
- `query_cache` — result cache with TTL
- `learning_events` — execution results for pattern learning

**Degrades gracefully**: if PostgreSQL is unavailable, all memory methods return empty results — query execution still works.

**Semantic search**: `generate_embedding()` (from `llm_client.py`) creates embeddings, pgvector `<=>` operator does cosine similarity search against HNSW indexes.

## Schema Discovery (`utils.py:SchemaManager`)

Three strategies in fallback order:
1. `.show table {table} schema as json` — most detailed (management query)
2. `{table} | getschema` — backup (data query)
3. `{table} | take 2` — last resort, infers types from sample values

Each strategy catches exceptions and falls through to the next.

## KQL Validation (`kql_validator.py`)

Pre-execution validation checks three things:
1. **Tables** — extracts table names, ensures schemas exist in memory
2. **Columns** — validates all referenced columns exist in schema (handles dynamic field access patterns)
3. **Operator syntax** — catches invalid negation patterns (`! =` → `!=`, `! contains` → `!contains`, `!has_any` → `!in`)

## Observability (`observability.py`)

Langfuse is a **hard dependency** (no graceful degradation). Uses Langfuse Python SDK v4:

- **LLM tracing**: `langfuse.openai` drop-in wrapper auto-traces all OpenAI calls (tokens, cost, latency). No `@observe` needed on LLM functions.
- **MCP tool tracing**: `@trace()` (wraps `@observe()`) on the 3 MCP tools + `_generate_kql_from_natural_language` + `_execute_kusto_query_sync`
- **Span metadata**: `update_trace()` calls `get_client().update_current_span()`. Metadata values must be `dict[str, str]` (v4 requirement).
- **Trace hierarchy**: Each MCP tool call = one root trace. Inner calls nest as child spans automatically.
- **Shutdown**: `flush()` calls `get_client().flush()` in `main()`'s `finally` block.
- **Env var mapping**: `LANGFUSE_BASEURL` (team convention) is mapped to `LANGFUSE_HOST` (SDK convention) at module load.
