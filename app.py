"""
app.py — Chat logic with PostgreSQL-backed persistent memory.

Changes from original:
  - _STORE (in-memory dict) is replaced by PostgreSQL via db.py
  - lc_history is still kept in-memory PER PROCESS for LangChain's
    RunnableWithMessageHistory (it needs a live object), but it is
    re-hydrated from the DB on first access within a process run.
  - raw_history reads/writes go to the DB.
  - Every message is embedded via embeddings.py and stored in the DB.
  - clear_session() deletes DB rows (not just a dict entry).
"""

import logging
import os

from dotenv import load_dotenv
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()  # loads .env before anything else reads env vars

from prompts import chat_prompt
import db
import embeddings as emb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "hf_******")
HF_MODEL  = os.getenv("HF_MODEL",  "Qwen/Qwen3-Coder-Next:novita")

llm = ChatOpenAI(
    model=HF_MODEL,
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
    temperature=0.2,
)

chat_chain = chat_prompt | llm

# ---------------------------------------------------------------------------
# In-process LangChain history cache (re-hydrated from DB on first use)
# ---------------------------------------------------------------------------

# session_id -> InMemoryChatMessageHistory  (only lives for the process lifetime)
_LC_CACHE: Dict[str, InMemoryChatMessageHistory] = {}


def _build_lc_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Load persisted messages from DB and build a fresh InMemoryChatMessageHistory.
    Called once per session per process start.
    """
    lc_hist = InMemoryChatMessageHistory()
    try:
        rows = db.get_messages(session_id)
        for row in rows:
            if row["role"] == "human":
                lc_hist.add_message(HumanMessage(content=row["content"]))
            else:
                lc_hist.add_message(AIMessage(content=row["content"]))
    except Exception as exc:
        logger.error("Failed to load history from DB for session '%s': %s", session_id, exc)
    return lc_hist


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Return the LangChain in-memory history for a session,
    hydrating from DB if this is the first access in the current process.
    """
    if session_id not in _LC_CACHE:
        _LC_CACHE[session_id] = _build_lc_history(session_id)
    return _LC_CACHE[session_id]


# ---------------------------------------------------------------------------
# LangChain runnable with memory
# ---------------------------------------------------------------------------

chat_with_memory = RunnableWithMessageHistory(
    chat_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _store_message(session_id: str, role: str, content: str) -> None:
    """Persist a message to DB and (async-ish) generate its embedding."""
    try:
        msg_id = db.insert_message(session_id, role, content)
    except Exception as exc:
        raise RuntimeError(f"DB insert failed: {exc}") from exc

    # Embedding generation is best-effort: never block or crash on failure.
    vector = emb.embed(content)
    if vector is not None:
        try:
            db.update_embedding(msg_id, vector)
        except Exception as exc:
            logger.warning("Could not store embedding for message %s: %s", msg_id, exc)


def chat(question: str, session_id: str = "default") -> str:
    question = (question or "").strip()
    if not question:
        return "اكتب رسالة أولاً."

    # Persist user message
    _store_message(session_id, "human", question)

    # Call LLM (uses in-memory LangChain history for context window)
    msg = chat_with_memory.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )
    answer = msg.content

    # Persist assistant message
    _store_message(session_id, "ai", answer)

    return answer


def get_raw_history(session_id: str = "default") -> list[dict]:
    """Return full chronological history from PostgreSQL."""
    try:
        rows = db.get_messages(session_id)
    except Exception as exc:
        raise RuntimeError(f"DB read failed: {exc}") from exc
    # Normalise to the original format: {"role": ..., "content": ...}
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def clear_session(session_id: str = "default") -> None:
    """Delete all DB records for the session and evict in-memory cache."""
    try:
        db.delete_session(session_id)
    except Exception as exc:
        raise RuntimeError(f"DB delete failed: {exc}") from exc
    # Reset in-memory LangChain history so the next chat starts fresh
    _LC_CACHE.pop(session_id, None)


# ---------------------------------------------------------------------------
# CLI (unchanged from original)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db.init_pool()

    session_id = input("Session ID (مثلاً student1): ").strip() or "default"

    while True:
        q = input("\nسؤال (أو اكتب exit): ").strip()
        if q.lower() == "exit":
            break

        if q.lower() == "/history":
            print("\n=== HISTORY ===")
            for i, item in enumerate(get_raw_history(session_id), 1):
                print(f"{i:02d}. [{item['role']}] {item['content']}")
            continue

        if q.lower() == "/clear":
            clear_session(session_id)
            print("تم مسح الذاكرة لهذه الجلسة.")
            continue

        print("\n---\n" + chat(q, session_id=session_id))
