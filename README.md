# Writr Backend

High-performance FastAPI backend for the **Writr** platform. Engineered for seamless dual-mode execution supporting both **Local Mode** (on-device Ollama LLMs & embedded SQLite VectorStore) and **Cloud Mode** (**Supabase** with `pgvector` & Cloud AI Providers like Gemini, OpenAI, Groq, and OpenRouter), orchestrated via **LangChain** and **LangGraph**.

---

## Features

- **Dual-Mode AI & Storage Architecture**:
  - **Local Mode**:
    - **Inference**: On-device LLMs and embeddings via Ollama (`nomic-embed-text`, `llama3`, `gemma`, etc.) with 100% data privacy.
    - **Vector Store**: Lightweight SQLite database storing float32 vectors in raw binary, matched with high-speed NumPy cosine similarity.
  - **Cloud Mode**:
    - **Inference**: Scalable cloud models (Google Gemini, OpenAI, Groq, OpenRouter).
    - **Vector Store & Database**: **Supabase** (PostgreSQL + `pgvector` extension) with remote vector indexing, metadata filtering, and `match_documents` RPC functions.
- **Embedded SQLite VectorStore**: Custom LangChain `VectorStore` implementation for local execution with zero external database setup.
- **Fast Vector Matching Engine**: Fully vectorized NumPy cosine similarity calculation (`match_reference_chunks`) with automatic L2 normalization, dimension checks, and division-by-zero safeguards.
- **Rate Limiting & Security**: Production-ready rate limiting powered by `SlowAPI` and configurable CORS middleware.
- **Modern Python Toolchain**: Built for Python `>=3.14` using the lightning-fast `uv` package manager.

---

## Project Structure

```text
write-backend/
├── main.py                      # FastAPI application entrypoint & middleware setup
├── pyproject.toml               # Project metadata & dependency definitions
├── uv.lock                      # Locked dependency tree
│
├── router/                      # API routing modules
│   ├── __init__.py              # Aggregates /api sub-routers
│   ├── local/                   # Local endpoints (Ollama, local SQLite RAG)
│   │   └── __init__.py
│   └── cloud/                   # Cloud endpoints (Supabase RAG, Gemini, OpenAI)
│       └── __init__.py
│
├── models/                      # Database & SQLModel schemas
│   ├── __init__.py
│   └── local/
│       ├── __init__.py
│       └── reference_chunk.py   # ReferenceChunks table (embeddings & metadata)
│
├── services/                    # Core business logic & LangChain integrations
│   └── local_vector_store.py    # Custom LangChain VectorStore for SQLite
│
└── utils/                       # Shared utilities & helpers
    ├── __init__.py
    ├── log.py                   # Uvicorn-compatible CLI logger
    └── matching.py              # Vectorized NumPy cosine similarity matcher
```

---

## Technology Stack

| Layer | Local Mode | Cloud Mode |
| :--- | :--- | :--- |
| **Database & Vector DB** | SQLite via [SQLModel](https://sqlmodel.tiangolo.com/) | [Supabase](https://supabase.com/) (PostgreSQL + `pgvector`) |
| **Vector Search Engine** | Vectorized [NumPy](https://numpy.org/) BLAS | Supabase `match_documents` RPC / IVFFlat / HNSW |
| **LLM & Embeddings** | [Ollama](https://ollama.ai/) (`nomic-embed-text`) | [Gemini](https://ai.google.dev/), [OpenAI](https://openai.com/), [Groq](https://groq.com/) |
| **Orchestration** | [LangChain](https://www.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/) | [LangChain](https://www.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/) |
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) | [FastAPI](https://fastapi.tiangolo.com/) |
| **Rate Limiting** | [SlowAPI](https://github.com/laurentS/slowapi) | [SlowAPI](https://github.com/laurentS/slowapi) |
| **Package Manager** | [uv](https://github.com/astral-sh/uv) (Python 3.14+) | [uv](https://github.com/astral-sh/uv) (Python 3.14+) |

---

## Getting Started

### Prerequisites

- **Python**: Version `3.14` or higher.
- **uv**: Modern Python package manager. Install via:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- *(For Local Mode)* **Ollama**:
  ```bash
  ollama serve
  ollama pull nomic-embed-text
  ```
- *(For Cloud Mode)* **Supabase Project**: A Supabase project with the `pgvector` extension enabled.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MaestroDev19/writr-backend.git
   cd write-backend
   ```

2. Sync dependencies:
   ```bash
   uv sync
   ```

3. Configure environment variables:
   Create a `.env` file in the project root:
   ```env
   # Server Configuration
   ENVIRONMENT=development
   PORT=8000

   # Local AI (Ollama)
   OLLAMA_ENDPOINT=http://localhost:11434
   OLLAMA_MODEL=llama3.1:8b
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text
   EMBEDDING_DIM=768

   # Cloud Storage & VectorStore (Supabase)
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_ANON_KEY=your-anon-key
   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

   # Cloud AI API Keys (Optional)
   GEMINI_API_KEY=your-gemini-key
   OPENAI_API_KEY=your-openai-key
   OPENROUTER_API_KEY=your-openrouter-key
   GROQ_API_KEY=your-groq-key
   ```

---

## Running the Server

Start the development server with live reload:

```bash
uv run fastapi dev main.py
```

Or with Uvicorn directly:

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API documentation:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Dual VectorStore Architecture

Writr enables users to toggle seamlessly between offline local computing and managed cloud persistence:

### 1. Local VectorStore (SQLite + NumPy)

- **Storage**: Raw `float32` binary vector blobs stored in SQLite (`ReferenceChunks` table).
- **Dimensionality**: Fixed to local model dimension (e.g. 768 for `nomic-embed-text` $\rightarrow$ 3072 bytes).
- **Matching Engine**: In-memory vectorized cosine similarity in `utils/matching.py`:
  ```python
  matrix = np.array([np.frombuffer(c.embedding, dtype=np.float32) for c in candidates])
  matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
  matrix_norms[matrix_norms == 0] = 1.0
  matrix = matrix / matrix_norms
  similarities = np.dot(matrix, query_vec)
  ```

### 2. Cloud VectorStore (Supabase + `pgvector`)

- **Storage**: Remote PostgreSQL table with native `vector` column type:
  ```sql
  create extension if not exists vector;

  create table reference_chunks (
      id uuid primary key default gen_random_uuid(),
      content text not null,
      metadata jsonb default '{}'::jsonb,
      embedding vector(1536) -- or 768 / model dimension
  );
  ```
- **Vector Search Function (RPC)**:
  ```sql
  create or replace function match_reference_chunks (
      query_embedding vector,
      match_count int default 5,
      filter jsonb default '{}'::jsonb
  ) returns table (
      id uuid,
      content text,
      metadata jsonb,
      similarity float
  ) language plpgsql as $$
  begin
      return query
      select
          reference_chunks.id,
          reference_chunks.content,
          reference_chunks.metadata,
          1 - (reference_chunks.embedding <=> query_embedding) as similarity
      from reference_chunks
      where reference_chunks.metadata @> filter
      order by reference_chunks.embedding <=> query_embedding
      limit match_count;
  end;
  $$;
  ```
- **LangChain Integration**: Connected via `SupabaseVectorStore` or Supabase Python client in `router/cloud/`.

---

## API Overview

| Method | Path | Target | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Core | System health check |
| `GET` | `/` | Core | Welcome root message |
| `GET/POST` | `/api/local/*` | Local | Ollama inference, local SQLite vector matching |
| `GET/POST` | `/api/cloud/*` | Cloud | Supabase `pgvector` search, Gemini / OpenAI inference |

---

## License

Private repository. All rights reserved.
