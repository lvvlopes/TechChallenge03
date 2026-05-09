"""
FASE 3 - assistant.py
Assistente medico interativo (modo conversacional) usando OpenAI + RAG.

Execute: python fase3_langchain/assistant.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import fix_windows_encoding
fix_windows_encoding()

from datetime import datetime
from config import OPENAI_API_KEY, OPENAI_MODEL, LLM_BACKEND
from fase3_langchain.graph import run_graph
from fase3_langchain.llm_backend import get_backend_info
from fase4_seguranca.logging_audit import log_interaction


def print_banner():
    info = get_backend_info()
    print("\n" + "=" * 60)
    print("  MedAssist - Assistente Medico IA")
    print(f"  Backend : {info['description']}")
    if LLM_BACKEND == "local":
        gpu_status = "GPU disponivel" if info.get("gpu_available") else "CPU (sem GPU)"
        print(f"  Hardware: {gpu_status}")
        if not info.get("adapters_exist"):
            print("  AVISO: Adaptadores LoRA nao encontrados em outputs/model/")
    print("=" * 60)
    print("  AVISO: Para fins informativos. Nao substitui consulta medica.")
    print("  Comandos: 'sair' para encerrar | 'historico' para ver log")
    print("=" * 60 + "\n")


def run_assistant():
    if LLM_BACKEND == "openai" and not OPENAI_API_KEY:
        print("ERRO: LLM_BACKEND=openai mas OPENAI_API_KEY nao configurada.")
        print("   1. Copie .env.example para .env")
        print("   2. Preencha: OPENAI_API_KEY=sk-...")
        print("   Ou mude para: LLM_BACKEND=local no .env")
        sys.exit(1)

    print_banner()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    history = []

    print("Como posso ajuda-lo hoje?\n")

    while True:
        try:
            user_input = input("Voce: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nEncerrando o MedAssist. Cuide-se!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("sair", "exit", "quit"):
            print("\nEncerrando o MedAssist. Cuide-se!")
            break

        if user_input.lower() in ("historico", "log"):
            print(f"\nHistorico da sessao ({len(history)} interacoes):")
            for i, (q, _) in enumerate(history, 1):
                print(f"  [{i}] {q[:80]}")
            print()
            continue

        print("\nProcessando...", end=" ", flush=True)
        try:
            result = run_graph(user_input)
            answer   = result["final_answer"]
            sources  = result["sources"]
            flags    = result["flags"]
            category = result["category"]

            print(f"\r{' ' * 20}\r", end="")
            print(f"\nMedAssist [{category}]:\n{answer}\n")

            log_interaction(
                session_id=session_id,
                question=user_input,
                answer=answer,
                sources=sources,
                flags=flags,
                category=category,
            )
            history.append((user_input, answer))

        except Exception as e:
            print(f"\rErro ao processar: {e}\n")
            print("Tente reformular sua pergunta.\n")


def main():
    run_assistant()


if __name__ == "__main__":
    main()
