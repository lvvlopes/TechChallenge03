"""
web_app.py — Interface web profissional do MedAssist

Servidor FastAPI com frontend completo embutido (HTML/CSS/JS).
Suporta LLM_BACKEND=openai ou servidor local via OPENAI_BASE_URL.

Uso:
    pip install fastapi uvicorn
    python web_app.py
    Acesse: http://localhost:8080
"""

import sys, json, time, uuid
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    LLM_BACKEND, OPENAI_MODEL, OPENAI_API_KEY,
    EMERGENCY_KEYWORDS, SAFETY_DISCLAIMER,
)

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Instale: pip install fastapi uvicorn")
    sys.exit(1)

app = FastAPI(title="MedAssist", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_graph_loaded = False
_run_graph = _get_backend_info = _log_interaction = None

def _load_project():
    global _graph_loaded, _run_graph, _get_backend_info, _log_interaction
    if _graph_loaded: return
    from fase3_langchain.graph import run_graph
    from fase3_langchain.llm_backend import get_backend_info
    from fase4_seguranca.logging_audit import log_interaction
    _run_graph = run_graph
    _get_backend_info = get_backend_info
    _log_interaction = log_interaction
    _graph_loaded = True

_session_config = {
    "backend": LLM_BACKEND, "api_key": OPENAI_API_KEY,
    "model": OPENAI_MODEL, "base_url": "",
}

HTML = open(Path(__file__).parent / "web_ui.html", encoding="utf-8").read() if (Path(__file__).parent / "web_ui.html").exists() else "<h1>web_ui.html not found</h1>"

class ChatRequest(BaseModel):
    message: str
    session_id: str = ""

class ConfigRequest(BaseModel):
    backend: str = "openai"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    base_url: str = ""

@app.get("/", response_class=HTMLResponse)
def index():
    return open(Path(__file__).parent / "web_ui.html", encoding="utf-8").read()

@app.get("/api/info")
def api_info():
    _load_project()
    info = _get_backend_info()
    info["model"] = _session_config["model"]
    return info

@app.post("/api/config")
def api_config(req: ConfigRequest):
    import os, importlib
    _session_config.update({"backend": req.backend, "api_key": req.api_key,
                            "model": req.model, "base_url": req.base_url})
    os.environ["LLM_BACKEND"] = req.backend
    os.environ["OPENAI_API_KEY"] = req.api_key
    os.environ["OPENAI_MODEL"] = req.model
    os.environ["OPENAI_BASE_URL"] = req.base_url
    import config as cfg_mod
    import fase3_langchain.llm_backend as be_mod
    importlib.reload(cfg_mod)
    importlib.reload(be_mod)
    return {"backend": req.backend, "model": req.model, "base_url": req.base_url}

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    _load_project()
    if not req.message.strip():
        return JSONResponse({"error": "Mensagem vazia."}, status_code=400)
    t0 = time.time()
    try:
        result = _run_graph(req.message.strip())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    elapsed = round(time.time() - t0, 1)
    try:
        _log_interaction(
            session_id=req.session_id or str(uuid.uuid4()),
            question=req.message, answer=result.get("final_answer", ""),
            category=result.get("category", ""), sources=result.get("sources", []),
            flags=result.get("flags", []),
        )
    except Exception:
        pass
    return {
        "answer": result.get("final_answer", "Sem resposta."),
        "category": result.get("category", ""),
        "sources": result.get("sources", []),
        "flags": result.get("flags", []),
        "backend": _session_config["backend"],
        "model": _session_config["model"],
        "elapsed": elapsed,
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MedAssist — Interface Web Profissional")
    print(f"  Backend : {LLM_BACKEND}  |  Modelo: {OPENAI_MODEL}")
    print(f"  Acesse  : http://localhost:8080")
    print("="*60 + "\n")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="warning",
        timeout_keep_alive=300,   # 5 min — Ollama em CPU pode ser lento
    )
