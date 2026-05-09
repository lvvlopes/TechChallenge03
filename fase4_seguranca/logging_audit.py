"""
FASE 4 — logging_audit.py
Sistema de logging estruturado para rastreamento e auditoria.

Execute: python fase4_seguranca/logging_audit.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from utils import fix_windows_encoding
fix_windows_encoding()

import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AUDIT_LOG_PATH, LOG_LEVEL

# Configura logger de auditoria
_audit_logger = logging.getLogger("medassist.audit")
_audit_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

if not _audit_logger.handlers:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _handler = logging.FileHandler(AUDIT_LOG_PATH, encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(message)s"))  # JSON raw
    _audit_logger.addHandler(_handler)


def log_interaction(
    session_id: str,
    question: str,
    answer: str,
    sources: list[str] = None,
    flags: list[str] = None,
    category: str = "consulta",
    user_id: str = "anon",
) -> dict:
    """
    Registra uma interação no log de auditoria em formato JSON estruturado.
    Retorna o dict logado.
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "session_id": session_id,
        "user_id": user_id,
        "category": category,
        "question_length": len(question),
        "question_preview": question[:100],
        "answer_length": len(answer),
        "answer_preview": answer[:200],
        "sources": sources or [],
        "flags": flags or [],
        "has_disclaimer": "Aviso Médico" in answer or "aviso" in answer.lower(),
    }
    _audit_logger.info(json.dumps(entry, ensure_ascii=False))
    return entry


def log_error(session_id: str, error: str, context: str = ""):
    """Registra um erro no log de auditoria."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "session_id": session_id,
        "level": "ERROR",
        "error": error,
        "context": context,
    }
    _audit_logger.error(json.dumps(entry, ensure_ascii=False))


def read_audit_log(n_last: int = 20) -> list[dict]:
    """Lê as últimas N entradas do log de auditoria."""
    if not AUDIT_LOG_PATH.exists():
        return []
    entries = []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries[-n_last:]


def audit_summary(entries: list[dict]) -> dict:
    """Gera sumário estatístico do log de auditoria."""
    if not entries:
        return {"total": 0}

    categories = {}
    flagged = 0
    for e in entries:
        cat = e.get("category", "desconhecido")
        categories[cat] = categories.get(cat, 0) + 1
        if e.get("flags"):
            flagged += 1

    return {
        "total_interactions": len(entries),
        "categories": categories,
        "flagged_interactions": flagged,
        "flag_rate": f"{flagged/len(entries)*100:.1f}%",
        "first": entries[0].get("timestamp", ""),
        "last": entries[-1].get("timestamp", ""),
    }


def demo():
    sep = "=" * 60
    print(f"\n{sep}")
    print("  📋 FASE 4 — Sistema de Logging e Auditoria")
    print(sep)

    session_id = f"demo_{datetime.now().strftime('%H%M%S')}"

    # Simula interações
    interactions = [
        ("Quais são os sintomas da diabetes?",
         "Os sintomas incluem sede, urina frequente e cansaço.",
         ["MedQuAD: diabetes tipo 2"], [], "consulta"),
        ("Como tratar hipertensão?",
         "O tratamento inclui mudanças no estilo de vida e medicamentos prescritos pelo médico.",
         ["MedQuAD: hipertensão"], ["AVISO: medicamento mencionado"], "consulta"),
        ("Sintomas de emergência cardíaca?",
         "🚨 EMERGÊNCIA DETECTADA...",
         ["Protocolo de Emergências"], ["EMERGÊNCIA DETECTADA"], "emergencia"),
    ]

    print(f"\n📝 Registrando {len(interactions)} interações de teste...")
    for q, a, sources, flags, category in interactions:
        entry = log_interaction(
            session_id=session_id, question=q, answer=a,
            sources=sources, flags=flags, category=category
        )
        print(f"   ✅ [{category}] {q[:50]}...")

    # Lê e mostra log
    print(f"\n📖 Log de auditoria ({AUDIT_LOG_PATH}):")
    entries = read_audit_log(n_last=len(interactions))
    for e in entries:
        ts = e.get("timestamp", "")[:19]
        cat = e.get("category", "")
        q = e.get("question_preview", "")[:60]
        flags = e.get("flags", [])
        flag_icon = "⚠️" if flags else "✅"
        print(f"   {flag_icon} [{ts}] [{cat}] {q}")

    # Sumário
    all_entries = read_audit_log(n_last=100)
    summary = audit_summary(all_entries)
    print(f"\n📊 Sumário do Log:")
    for k, v in summary.items():
        print(f"   {k}: {v}")

    print(f"\n{sep}")
    print("  ✅ Fase 4.2 — Logging de Auditoria validado!")
    print(sep)


if __name__ == "__main__":
    demo()
