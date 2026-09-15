# Writr Backend

High-performance FastAPI backend for the **Writr** platform. Engineered for seamless dual-mode execution supporting both **Local LLMs & Embeddings** (via Ollama) and **Cloud AI Providers** (Gemini, OpenAI, Groq, OpenRouter), powered by **LangChain**, **LangGraph**, and a lightweight **SQLite-backed VectorStore**.

---

## Features

- **Dual-Mode AI Architecture**:
  - **Local**: On-device inference via Ollama (`nomic-embed-text`, `llama3`, `gemma`, etc.) with zero cloud data transmission.
  - **Cloud**: Scalable external providers (Gemini, OpenAI, OpenRouter, Groq).
- **Embedded SQLite VectorStore**: Custom LangChain `VectorStore` implementation using SQLModel and raw binary storage (`float32` BLAS vectors) with zero external database dependencies.
- **Fast Vector Matching Engine**: Fully vectorized NumPy cosine similarity calculation (`match_reference_chunks`) with automatic L2 normalization, dimension checks, and division-by-zero safeguards.
- **Rate Limiting & Security**: Production-ready rate limiting powered by `SlowAPI` and configurable CORS middleware.
- **Modern Python Toolchain**: Developed for Python `>=3.14` using the lightning-fast `uv` package and project manager.

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
│   ├── local/                   # Local endpoints (Ollama, local DB)
│   │   └── __init__.py
│   └── cloud/                   # Cloud endpoints (Gemini, OpenAI, etc.)
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

| Layer | Technologies |
| :--- | :--- |
| **Framework** | [FastAPI](https://fastapi.tiangolo.com/), [Starlette](https://www.starlette.io/) |
| **Server** | [Uvicorn](https://www.uvicorn.org/) |
| **Package Manager** | [uv](https://github.com/astral-sh/uv) |
| **AI Orchestration** | [LangChain](https://www.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/) |
| **Database / ORM** | [SQLModel](https://sqlmodel.tiangolo.com/), [SQLAlchemy](https://www.sqlalchemy.org/), [SQLite](https://www.sqlite.org/) |
| **Numerical Computing** | [NumPy](https://numpy.org/) |
| **Rate Limiting** | [SlowAPI](https://github.com/laurentS/slowapi) |

---

## Getting Started

### Prerequisites

- **Python**: Version `3.14` or higher.
- **uv**: Modern Python package manager. Install via:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- *(Optional for Local Mode)* **Ollama**: Installed and running locally:
  ```bash
  ollama serve
  ollama pull nomic-embed-text
  ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MaestroDev19/writr-backend.git
   cd write-backend
   ```

2. Create virtual environment and install dependencies:
   ```bash
   uv sync
   ```

3. Configure environment variables:
   Create a `.env` file in the project root:
   ```env
   # General
   ENVIRONMENT=development
   PORT=8000

   # Local AI (Ollama)
   OLLAMA_ENDPOINT=http://localhost:11434
   OLLAMA_MODEL=llama3.1:8b
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text
   EMBEDDING_DIM=768

   # Cloud AI API Keys (Optional)
   GEMINI_API_KEY=
   OPENAI_API_KEY=
   OPENROUTER_API_KEY=
   ```

---

## Running the Server

Start the development server with live reload:

```bash
uv run fastapi dev main.py
```

Alternatively, run with Uvicorn directly:

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Once running, interactive documentation is accessible at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Core Architecture Deep Dive

### 1. SQLite Vector Storage (`ReferenceChunks`)

Instead of requiring external vector databases (such as Qdrant, Pinecone, or pgvector), `write-backend` serializes raw float vectors as compact binary blobs (`float32` byte arrays) inside SQLite.

- **Column**: `embedding: bytes = Field(sa_column=Column("embedding", LargeBinary))`
- **Metadata**: `metadata_json: Dict[str, Any] = Field(default={}, sa_column=Column("metadata", JSON))`
- **Dimensionality**: For `nomic-embed-text` (768 dimensions), each stored embedding occupies exactly $768 \times 4 = 3072$ bytes.

### 2. High-Performance NumPy Cosine Similarity

Vector retrieval in `utils/matching.py` executes in-memory NumPy operations:

```python
# 1. Decode float32 arrays from candidate binary buffers
matrix = np.array([np.frombuffer(c.embedding, dtype=np.float32) for c in candidates])

# 2. Vectorized L2 Normalization with division-by-zero protection
matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
matrix_norms[matrix_norms == 0] = 1.0
matrix = matrix / matrix_norms

# 3. Normalized Query Dot Product
similarities = np.dot(matrix, query_vec)
top_indices = np.argsort(similarities)[::-1][:match_count]
```

### 3. Custom LangChain `LocalVectorStore`

The `LocalVectorStore` service implements `langchain_core.vectorstores.VectorStore`, allowing seamless integration with LangChain chains and LangGraph agents:

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
```

---

## API Overview

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/` | Root welcome endpoint |
| `GET` | `/api/local/*` | Local Ollama inference & local RAG routes |
| `GET` | `/api/cloud/*` | Cloud provider generation & critique routes |

---

## License

Private repository. All rights reserved.
