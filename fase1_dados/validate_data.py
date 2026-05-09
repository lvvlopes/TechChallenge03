"""
FASE 1 — validate_data.py
Validação de qualidade dos splits de treino e validação.

Execute: python fase1_dados/validate_data.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from utils import fix_windows_encoding
fix_windows_encoding()

import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_root = Path(__file__).parent.parent
TRAIN_PATH = _root / "data" / "train.jsonl"
VAL_PATH   = _root / "data" / "val.jsonl"
REPORT_PATH = _root / "outputs" / "data_quality_report.txt"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_texts(records: list[dict]) -> tuple[list[str], list[str]]:
    questions, answers = [], []
    for r in records:
        msgs = r.get("messages", [])
        q = next((m["content"] for m in msgs if m["role"] == "user"), "")
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        questions.append(q)
        answers.append(a)
    return questions, answers


def check_leakage(train: list[dict], val: list[dict]) -> int:
    """Verifica se alguma pergunta do val está no treino (data leakage)."""
    train_q = set()
    for r in train:
        msgs = r.get("messages", [])
        q = next((m["content"] for m in msgs if m["role"] == "user"), "")
        train_q.add(q.strip().lower())

    leaked = 0
    for r in val:
        msgs = r.get("messages", [])
        q = next((m["content"] for m in msgs if m["role"] == "user"), "")
        if q.strip().lower() in train_q:
            leaked += 1
    return leaked


def stats(texts: list[str]) -> dict:
    lens = [len(t.split()) for t in texts]
    return {
        "count": len(lens),
        "min": min(lens), "max": max(lens),
        "avg": sum(lens) / len(lens) if lens else 0,
    }


def validate(train: list[dict], val: list[dict]) -> dict:
    train_q, train_a = extract_texts(train)
    val_q, val_a = extract_texts(val)

    issues = []
    if len(train) == 0:
        issues.append("CRÍTICO: Split de treino está vazio!")
    if len(val) == 0:
        issues.append("CRÍTICO: Split de validação está vazio!")

    leaked = check_leakage(train, val)
    if leaked > 0:
        issues.append(f"AVISO: {leaked} perguntas do val existem no treino (data leakage).")

    # Registros com campos vazios
    empty_q_train = sum(1 for q in train_q if not q.strip())
    empty_a_train = sum(1 for a in train_a if not a.strip())
    if empty_q_train:
        issues.append(f"AVISO: {empty_q_train} perguntas vazias no treino.")
    if empty_a_train:
        issues.append(f"AVISO: {empty_a_train} respostas vazias no treino.")

    return {
        "train": {"records": len(train), "questions": stats(train_q), "answers": stats(train_a)},
        "val":   {"records": len(val),   "questions": stats(val_q),   "answers": stats(val_a)},
        "leakage": leaked,
        "issues": issues,
    }


def print_report(result: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  📋 RELATÓRIO DE QUALIDADE DOS DADOS")
    lines.append("=" * 60)

    for split_name in ["train", "val"]:
        s = result[split_name]
        lines.append(f"\n{'🚂 Treino' if split_name == 'train' else '🔬 Validação'}:")
        lines.append(f"   Registros:  {s['records']:,}")
        lines.append(f"   Perguntas — min:{s['questions']['min']} max:{s['questions']['max']} avg:{s['questions']['avg']:.1f} palavras")
        lines.append(f"   Respostas  — min:{s['answers']['min']} max:{s['answers']['max']} avg:{s['answers']['avg']:.1f} palavras")

    lines.append(f"\n🔄 Data Leakage: {result['leakage']} amostras sobrepostas")

    if result["issues"]:
        lines.append(f"\n⚠️  Problemas encontrados ({len(result['issues'])}):")
        for issue in result["issues"]:
            lines.append(f"   • {issue}")
    else:
        lines.append("\n✅ Nenhum problema crítico encontrado.")

    lines.append("\n" + "=" * 60)
    status = "✅ Fase 1.3 — Validação APROVADA!" if not any("CRÍTICO" in i for i in result["issues"]) else "❌ Fase 1.3 — Validação REPROVADA!"
    lines.append(f"  {status}")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    print("\n🔍 Verificando splits gerados...")

    for path in [TRAIN_PATH, VAL_PATH]:
        if not path.exists():
            print(f"❌ ERRO: Arquivo não encontrado: {path}")
            print("   Execute primeiro: python fase1_dados/preprocess.py")
            sys.exit(1)

    train = load_jsonl(TRAIN_PATH)
    val   = load_jsonl(VAL_PATH)
    print(f"   Treino: {len(train)} | Validação: {len(val)}")

    result = validate(train, val)
    report = print_report(result)
    print(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📄 Relatório salvo em: {REPORT_PATH}")


if __name__ == "__main__":
    main()
