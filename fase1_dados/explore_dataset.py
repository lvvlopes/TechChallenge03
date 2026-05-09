"""
FASE 1 — explore_dataset.py
Análise exploratória do dataset MedQuAD PT-BR.

Execute: python fase1_dados/explore_dataset.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from utils import fix_windows_encoding
fix_windows_encoding()

import json
import shutil
from pathlib import Path
from collections import Counter

# Adiciona raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Copia dataset para pasta data/ se ainda não estiver lá
SOURCE = Path(__file__).parent.parent.parent / "medquad_ptbr.jsonl"
DATA_DIR = Path(__file__).parent.parent / "data"
DATASET_PATH = DATA_DIR / "medquad_ptbr.jsonl"

if SOURCE.exists() and not DATASET_PATH.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(SOURCE, DATASET_PATH)

# Se o arquivo não existir em nenhum lugar, tenta o caminho configurado
if not DATASET_PATH.exists():
    from config import DATASET_PATH

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class Console:
        def print(self, *a, **kw): print(*a)
    console = Console()


def load_dataset(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_parts(record: dict) -> tuple[str, str]:
    """Extrai pergunta (user) e resposta (assistant) do registro."""
    messages = record.get("messages", [])
    question = next((m["content"] for m in messages if m["role"] == "user"), "")
    answer   = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    return question, answer


def analyze(records: list[dict]) -> dict:
    questions, answers = [], []
    for r in records:
        q, a = extract_parts(r)
        questions.append(q)
        answers.append(a)

    q_lens = [len(q.split()) for q in questions]
    a_lens = [len(a.split()) for a in answers]

    # Palavras mais frequentes nas perguntas (aproximação de categorias)
    all_words = " ".join(questions).lower().split()
    freq = Counter(w for w in all_words if len(w) > 4)

    # Primeiras palavras das perguntas (padrão de tipo de pergunta)
    starters = Counter(q.split()[0].lower() if q.split() else "" for q in questions)

    return {
        "total": len(records),
        "q_len_min": min(q_lens), "q_len_max": max(q_lens),
        "q_len_avg": sum(q_lens) / len(q_lens),
        "a_len_min": min(a_lens), "a_len_max": max(a_lens),
        "a_len_avg": sum(a_lens) / len(a_lens),
        "top_words": freq.most_common(15),
        "top_starters": starters.most_common(10),
        "sample_questions": questions[:5],
        "sample_answers": answers[:5],
    }


def print_report(stats: dict):
    sep = "=" * 60
    print(f"\n{sep}")
    print("  📊 ANÁLISE EXPLORATÓRIA — MedQuAD PT-BR")
    print(sep)
    print(f"\n✅ Total de amostras: {stats['total']:,}")
    print(f"\n📝 Perguntas (nº de palavras):")
    print(f"   Min: {stats['q_len_min']} | Max: {stats['q_len_max']} | Média: {stats['q_len_avg']:.1f}")
    print(f"\n💬 Respostas (nº de palavras):")
    print(f"   Min: {stats['a_len_min']} | Max: {stats['a_len_max']} | Média: {stats['a_len_avg']:.1f}")

    print(f"\n🔤 Palavras mais frequentes nas perguntas:")
    for word, count in stats["top_words"]:
        print(f"   {word:<20} {count:>5}")

    print(f"\n🗂️  Tipos de perguntas (palavra inicial):")
    for starter, count in stats["top_starters"]:
        print(f"   {starter:<20} {count:>5}")

    print(f"\n📋 Exemplos de pares pergunta/resposta:")
    for i, (q, a) in enumerate(zip(stats["sample_questions"], stats["sample_answers"]), 1):
        print(f"\n  [{i}] Pergunta: {q[:120]}{'...' if len(q) > 120 else ''}")
        print(f"       Resposta: {a[:200]}{'...' if len(a) > 200 else ''}")

    print(f"\n{sep}")
    print("  ✅ Fase 1.1 — Exploração concluída com sucesso!")
    print(sep)


def main():
    print(f"\n🔍 Carregando dataset de: {DATASET_PATH}")
    if not Path(DATASET_PATH).exists():
        print(f"❌ ERRO: Dataset não encontrado em {DATASET_PATH}")
        print("   Verifique se o arquivo medquad_ptbr.jsonl está em data/")
        sys.exit(1)

    records = load_dataset(Path(DATASET_PATH))
    print(f"✅ {len(records)} registros carregados.\n")

    stats = analyze(records)
    print_report(stats)

    # Salva relatório simples
    output = Path(__file__).parent.parent / "outputs" / "explore_report.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(f"Total de amostras: {stats['total']}\n")
        f.write(f"Perguntas - min:{stats['q_len_min']} max:{stats['q_len_max']} avg:{stats['q_len_avg']:.1f}\n")
        f.write(f"Respostas - min:{stats['a_len_min']} max:{stats['a_len_max']} avg:{stats['a_len_avg']:.1f}\n")
        f.write("\nTop palavras:\n")
        for w, c in stats["top_words"]:
            f.write(f"  {w}: {c}\n")
    print(f"\n📄 Relatório salvo em: {output}")


if __name__ == "__main__":
    main()
