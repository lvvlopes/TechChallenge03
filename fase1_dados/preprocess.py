"""
FASE 1 — preprocess.py
Limpeza, anonimização e split do dataset MedQuAD PT-BR.

Execute: python fase1_dados/preprocess.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from utils import fix_windows_encoding
fix_windows_encoding()

import json
import re
import shutil
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# Localiza o dataset
_project_root = Path(__file__).parent.parent
_source_candidates = [
    _project_root / "data" / "medquad_ptbr.jsonl",
    _project_root.parent / "medquad_ptbr.jsonl",
]
DATASET_PATH = next((p for p in _source_candidates if p.exists()), _source_candidates[0])
TRAIN_PATH   = _project_root / "data" / "train.jsonl"
VAL_PATH     = _project_root / "data" / "val.jsonl"
VAL_SPLIT    = 0.1

# Padrões de anonimização
ANON_PATTERNS = [
    (r"\bDr\.?\s+[A-Z][a-zÀ-ú]+\b", "[MÉDICO]"),
    (r"\bDra\.?\s+[A-Z][a-zÀ-ú]+\b", "[MÉDICA]"),
    (r"\bPaciente\s+[A-Z][a-zÀ-ú]+\b", "[PACIENTE]"),
    (r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "[CPF]"),
    (r"\b\d{2}/\d{2}/\d{4}\b", "[DATA]"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\b\(\d{2}\)\s*\d{4,5}-\d{4}\b", "[TELEFONE]"),
]


def clean_text(text: str) -> str:
    """Limpeza básica do texto."""
    # Remove espaços múltiplos
    text = re.sub(r" {2,}", " ", text)
    # Remove quebras de linha excessivas
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove caracteres de controle (mantém \n e \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def anonymize_text(text: str) -> str:
    """Aplica padrões de anonimização."""
    for pattern, replacement in ANON_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def is_valid_record(record: dict) -> tuple[bool, Optional[str]]:
    """Valida se o registro tem a estrutura esperada e conteúdo mínimo."""
    messages = record.get("messages", [])
    if not messages:
        return False, "sem campo 'messages'"

    roles = [m.get("role", "") for m in messages]
    if "user" not in roles:
        return False, "sem mensagem 'user'"
    if "assistant" not in roles:
        return False, "sem mensagem 'assistant'"

    question = next((m["content"] for m in messages if m["role"] == "user"), "")
    answer   = next((m["content"] for m in messages if m["role"] == "assistant"), "")

    if len(question.strip()) < 5:
        return False, "pergunta muito curta"
    if len(answer.strip()) < 10:
        return False, "resposta muito curta"

    return True, None


def process_record(record: dict) -> dict:
    """Processa um único registro: limpa e anonimiza."""
    new_messages = []
    for msg in record.get("messages", []):
        content = msg.get("content", "")
        content = clean_text(content)
        content = anonymize_text(content)
        new_messages.append({"role": msg["role"], "content": content})
    return {"messages": new_messages}


def load_raw(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_dataset(records: list[dict], val_ratio: float) -> tuple[list, list]:
    import random
    random.seed(42)
    shuffled = records.copy()
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    return shuffled[n_val:], shuffled[:n_val]  # train, val


def main():
    sep = "=" * 60
    print(f"\n{sep}")
    print("  🔧 FASE 1 — Pré-processamento dos Dados")
    print(sep)

    # Verifica dataset
    if not DATASET_PATH.exists():
        print(f"❌ ERRO: Dataset não encontrado em {DATASET_PATH}")
        sys.exit(1)

    print(f"\n📂 Carregando: {DATASET_PATH}")
    raw = load_raw(DATASET_PATH)
    print(f"   {len(raw)} registros brutos carregados.")

    # Valida e filtra
    print("\n🔍 Validando registros...")
    valid, invalid_reasons = [], []
    for r in raw:
        ok, reason = is_valid_record(r)
        if ok:
            valid.append(r)
        else:
            invalid_reasons.append(reason)

    print(f"   ✅ Válidos:  {len(valid)}")
    print(f"   ❌ Inválidos: {len(invalid_reasons)}")
    if invalid_reasons:
        from collections import Counter
        for reason, count in Counter(invalid_reasons).most_common():
            print(f"      - {reason}: {count}x")

    # Processa (limpa + anonimiza)
    print("\n🧹 Aplicando limpeza e anonimização...")
    processed = [process_record(r) for r in valid]
    print(f"   ✅ {len(processed)} registros processados.")

    # Split treino/validação
    print(f"\n✂️  Dividindo dataset (val={VAL_SPLIT*100:.0f}%)...")
    train, val = split_dataset(processed, VAL_SPLIT)
    print(f"   Treino:    {len(train)} amostras → {TRAIN_PATH}")
    print(f"   Validação: {len(val)} amostras  → {VAL_PATH}")

    # Salva
    save_jsonl(train, TRAIN_PATH)
    save_jsonl(val, VAL_PATH)

    # Salva amostra para inspeção
    sample_path = _project_root / "data" / "processed" / "sample_processed.jsonl"
    save_jsonl(processed[:10], sample_path)

    print(f"\n✅ Splits salvos com sucesso!")
    print(f"   Amostra de inspeção: {sample_path}")
    print(f"\n{sep}")
    print("  ✅ Fase 1.2 — Pré-processamento concluído!")
    print(sep)


if __name__ == "__main__":
    main()
