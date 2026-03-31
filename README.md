# 🤖 Arabic LangChain Chatbot

A production-ready, Arabic-first conversational AI backend powered by **LangChain**, **FastAPI**, and **PostgreSQL with pgvector**. Features persistent multi-session memory, semantic similarity search over chat history, and a Gradio UI — all backed by Hugging Face–hosted LLMs.

---

## ✨ Features

- 🧠 **Persistent Memory** — Chat history stored in PostgreSQL, re-hydrated across process restarts
- 🔍 **Semantic Search** — Vector similarity search over past messages via `pgvector` + `sentence-transformers`
- 🌐 **REST API** — Clean FastAPI endpoints for chat, history, search, and session management
- 🖥️ **Gradio UI** — Browser-based chat interface with session control and semantic search tab
- 🌍 **Multilingual Embeddings** — Default model supports Arabic + English out of the box
- ⚡ **LangChain Runnable Pipeline** — Prompt templating and message history managed via LangChain v0.2+

---

## 🗂️ Project Structure

```
.
├── api.py                  # FastAPI application — all HTTP endpoints
├── app.py                  # Core chat logic, LangChain pipeline, session management
├── db.py                   # PostgreSQL + pgvector persistence layer (connection pool, CRUD)
├── embeddings.py           # Lazy-loaded sentence-transformers embedding pipeline
├── prompts.py              # LangChain ChatPromptTemplate (system prompt + history placeholder)
├── ui_gradio_api.py        # Gradio frontend — connects to the FastAPI backend
├── migrate.sql             # One-time SQL migration: pgvector extension + schema
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (not committed — see .env.example)
```

---

## 🧱 Architecture

```
┌────────────────────────────────────────────────────────┐
│                     Client Layer                       │
│   Gradio UI (ui_gradio_api.py)  │  Direct API Calls   │
└─────────────────────┬──────────────────────────────────┘
                      │ HTTP (REST)
┌─────────────────────▼──────────────────────────────────┐
│               FastAPI Server  (api.py)                 │
│  POST /chat  GET /history/:id  GET /history/:id/search │
│  POST /clear                   GET /health             │
└──────┬────────────────────────────────┬────────────────┘
       │                                │
┌──────▼──────────────┐    ┌────────────▼───────────────┐
│   Chat Logic        │    │   Embedding Pipeline       │
│   (app.py)          │    │   (embeddings.py)          │
│                     │    │   sentence-transformers    │
│  LangChain Runnable │    │   paraphrase-multilingual  │
│  + Message History  │    │   MiniLM-L12-v2 (384-dim) │
└──────┬──────────────┘    └────────────┬───────────────┘
       │                                │
┌──────▼────────────────────────────────▼───────────────┐
│             PostgreSQL + pgvector  (db.py)             │
│                  chat_messages table                   │
│   id │ session_id │ role │ content │ embedding │ ts   │
│                                                        │
│  Indexes: session_id (B-Tree) · embedding (IVFFlat)   │
└────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────┐
│           Hugging Face LLM Router                      │
│       Qwen/Qwen3-Coder  (via OpenAI-compat API)        │
└────────────────────────────────────────────────────────┘
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ with the [`pgvector`](https://github.com/pgvector/pgvector) extension installed
- A [Hugging Face](https://huggingface.co/) account and API token

### 1. Clone & install dependencies

```bash
git clone https://github.com/your-username/arabic-langchain-chatbot.git
cd arabic-langchain-chatbot

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

```env
# .env
HF_TOKEN=hf_your_token_here
HF_MODEL=Qwen/Qwen3-Coder-Next:novita

PG_HOST=127.0.0.1
PG_PORT=5432
PG_DB=chatbot
PG_USER=postgres
PG_PASSWORD=your_password

EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### 3. Prepare the database

```bash
# Create the database
psql -U postgres -c "CREATE DATABASE chatbot;"

# Run the migration
psql -U postgres -d chatbot -f migrate.sql
```

> The app will also auto-create the schema on startup via `db._create_schema()`.

### 4. Start the API server

```bash
uvicorn api:api --reload --host 127.0.0.1 --port 8000
```

### 5. Launch the Gradio UI *(optional)*

In a separate terminal:

```bash
python ui_gradio_api.py
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Send a message and receive a reply |
| `GET` | `/history/{session_id}` | Full chronological chat history |
| `GET` | `/history/{session_id}/search?query=&k=` | Semantic similarity search over history |
| `POST` | `/clear` | Delete all messages for a session |

### Example — Send a message

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "student1", "question": "ما هو التعلم الآلي؟"}'
```

```json
{
  "session_id": "student1",
  "answer": "التعلم الآلي هو فرع من فروع الذكاء الاصطناعي..."
}
```

### Example — Semantic search

```bash
curl "http://127.0.0.1:8000/history/student1/search?query=neural+networks&k=3"
```

---

## 🗄️ Database Schema

```sql
CREATE TABLE chat_messages (
    id          BIGSERIAL    PRIMARY KEY,
    session_id  TEXT         NOT NULL,
    role        TEXT         NOT NULL CHECK (role IN ('human', 'ai')),
    content     TEXT         NOT NULL,
    embedding   vector(384),           -- paraphrase-multilingual-MiniLM-L12-v2
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
```

| Index | Type | Purpose |
|-------|------|---------|
| `idx_chat_messages_session_id` | B-Tree | Fast per-session message lookups |
| `idx_chat_messages_embedding` | IVFFlat (cosine) | Approximate nearest-neighbour search |

> To use a larger embedding model (e.g. 768-dim), update both `EMBEDDING_MODEL` in `.env` and the `vector(384)` column type in the migration.

---

## 🧩 Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `api.py` | FastAPI app, route definitions, request/response schemas (Pydantic) |
| `app.py` | LangChain pipeline, in-process history cache, public `chat()` / `clear_session()` / `get_raw_history()` functions |
| `db.py` | Connection pool lifecycle, all SQL queries, pgvector insert/search |
| `embeddings.py` | Lazy model loading, `embed(text) → list[float] \| None` |
| `prompts.py` | System prompt and `ChatPromptTemplate` definition |
| `ui_gradio_api.py` | Gradio frontend; talks to the API via HTTP only — no direct DB/LLM access |
| `migrate.sql` | Idempotent one-time database migration |

---

## 🛠️ Development

### Run tests *(coming soon)*

```bash
pytest tests/
```

### Linting

```bash
ruff check .
```

### Environment variables reference

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | Hugging Face API token (**required**) |
| `HF_MODEL` | `Qwen/Qwen3-Coder-Next:novita` | Model ID on HF router |
| `PG_HOST` | `127.0.0.1` | PostgreSQL host |
| `PG_PORT` | `5432` | PostgreSQL port |
| `PG_DB` | `chatbot` | Database name |
| `PG_USER` | `postgres` | Database user |
| `PG_PASSWORD` | `password` | Database password |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformers model |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Hugging Face Router → Qwen3 (OpenAI-compatible) |
| Orchestration | LangChain v0.2 (`RunnableWithMessageHistory`) |
| API | FastAPI + Uvicorn |
| UI | Gradio 4/5 (version-adaptive) |
| Database | PostgreSQL 14+ + pgvector |
| Embeddings | `sentence-transformers` — MiniLM-L12-v2 (384-dim) |
| Driver | psycopg2 (threaded connection pool) |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
