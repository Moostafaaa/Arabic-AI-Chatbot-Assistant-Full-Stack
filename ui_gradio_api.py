# run:  python ui_gradio_api.py

import requests
import gradio as gr

API_BASE = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Detect Gradio major version once at startup
# Gradio 4.x : Chatbot history = list of (user, assistant) tuples
# Gradio 5.x : Chatbot history = list of {"role":..., "content":...} dicts
# ---------------------------------------------------------------------------
_GR_MAJOR = int(gr.__version__.split(".")[0])
_USE_DICT_FORMAT = _GR_MAJOR >= 5   # True → dicts, False → tuples

print(f"Gradio {gr.__version__} detected — using {'dict' if _USE_DICT_FORMAT else 'tuple'} message format.")


def _make_chatbot():
    """Create a Chatbot component compatible with the installed Gradio version."""
    return gr.Chatbot(label="Chat")

def _append(history, user_msg, assistant_msg):
    """Append a turn to history in the correct format."""
    history = history or []
    if _USE_DICT_FORMAT:
        history.append({"role": "user",      "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})
    else:
        history.append((user_msg, assistant_msg))
    return history


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_chat(message, history, session_id):
    message = (message or "").strip()
    if not message:
        return "", history

    try:
        r = requests.post(
            f"{API_BASE}/chat",
            json={"session_id": session_id, "question": message},
            timeout=120,
        )
        r.raise_for_status()
        answer = r.json()["answer"]
    except Exception as e:
        answer = f"API error: {e}"

    return "", _append(history, message, answer)


def clear_and_update(sid):
    try:
        r = requests.post(f"{API_BASE}/clear", json={"session_id": sid}, timeout=30)
        r.raise_for_status()
        return [], f"✅ Cleared session: {sid}"
    except Exception as e:
        return gr.update(), f"❌ API error: {e}"


def fetch_history(sid):
    try:
        r = requests.get(f"{API_BASE}/history/{sid}", timeout=30)
        r.raise_for_status()
        items = r.json().get("history", [])
        lines = [f"{i:02d}. [{it.get('role')}] {it.get('content')}"
                 for i, it in enumerate(items, 1)]
        return "\n".join(lines) if lines else "(No history)"
    except Exception as e:
        return f"❌ API error: {e}"


def semantic_search(sid, query, k):
    if not query.strip():
        return "Enter a search query first."
    try:
        r = requests.get(
            f"{API_BASE}/history/{sid}/search",
            params={"query": query, "k": int(k)},
            timeout=30,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return "(No results found)"
        lines = []
        for i, res in enumerate(results, 1):
            sim = res.get("similarity", 0)
            lines.append(
                f"{i:02d}. [{res['role']}] (similarity={sim:.3f})\n"
                f"    {res['content']}\n"
                f"    @ {res.get('created_at', '')}"
            )
        return "\n\n".join(lines)
    except Exception as e:
        return f"❌ API error: {e}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Chat via FastAPI") as demo:
    gr.Markdown("## 🤖 Chatbot UI  ·  FastAPI + PostgreSQL backend")

    session_id = gr.Textbox(label="Session ID", value="student1")

    with gr.Tab("💬 Chat"):
        chatbot     = _make_chatbot()
        msg         = gr.Textbox(label="Message", placeholder="اكتب رسالتك هنا…")
        status      = gr.Markdown("")

        with gr.Row():
            clear_btn = gr.Button("🗑 Clear History")
            show_btn  = gr.Button("📜 Show History")

        history_box = gr.Textbox(label="Session History (raw)", lines=10)

        msg.submit(api_chat,        inputs=[msg, chatbot, session_id], outputs=[msg, chatbot])
        clear_btn.click(clear_and_update, inputs=[session_id],         outputs=[chatbot, status])
        show_btn.click(fetch_history,     inputs=[session_id],         outputs=[history_box])

    with gr.Tab("🔍 Semantic Search"):
        gr.Markdown(
            "Search past messages by meaning (requires embedding model to be running)."
        )
        search_query   = gr.Textbox(label="Search Query")
        search_k       = gr.Slider(1, 20, value=5, step=1, label="Top-k results")
        search_btn     = gr.Button("Search")
        search_results = gr.Textbox(label="Results", lines=15)

        search_btn.click(
            semantic_search,
            inputs=[session_id, search_query, search_k],
            outputs=[search_results],
        )

demo.launch()