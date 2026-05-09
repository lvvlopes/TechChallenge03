"""
FASE 3 - graph.py
Fluxo de decisao automatizado com LangGraph + OpenAI.

Nos do grafo:
  entrada -> triagem -> [emergencia | busca_rag] -> geracao -> validacao -> saida

Execute: python fase3_langchain/graph.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import fix_windows_encoding
fix_windows_encoding()

from typing import TypedDict, Optional

from config import (
    EMERGENCY_KEYWORDS, PRESCRIPTION_KEYWORDS,
    SAFETY_DISCLAIMER, OPENAI_API_KEY, OPENAI_MODEL,
    SYSTEM_PROMPT, TEMPERATURE, TOP_K_RETRIEVAL,
    VECTOR_STORE_PATH, EMBEDDING_MODEL, LLM_BACKEND,
)
from fase3_langchain.llm_backend import generate as llm_generate, get_backend_info

_root = Path(__file__).parent.parent


# ─────────────────────────────────────────────
# Estado do Grafo
# ─────────────────────────────────────────────
class MedState(TypedDict):
    question: str
    category: Optional[str]
    retrieved_docs: list[str]
    raw_answer: str
    final_answer: str
    sources: list[str]
    flags: list[str]
    log: list[str]


# ─────────────────────────────────────────────
# Nos do Grafo
# ─────────────────────────────────────────────

def node_entrada(state: MedState) -> MedState:
    q = state["question"].strip()
    state["log"].append(f"[ENTRADA] {q[:80]}")
    state["question"] = q
    return state


def node_triagem(state: MedState) -> MedState:
    q = state["question"].lower()
    if any(kw in q for kw in EMERGENCY_KEYWORDS):
        state["category"] = "emergencia"
        state["flags"].append("EMERGENCIA DETECTADA")
    elif any(kw in q for kw in ["ola", "bom dia", "oi ", "tudo bem", "tchau"]):
        state["category"] = "fora_escopo"
    else:
        state["category"] = "consulta"
    state["log"].append(f"[TRIAGEM] categoria: {state['category']}")
    return state


def node_emergencia(state: MedState) -> MedState:
    state["raw_answer"] = (
        "SITUACAO DE EMERGENCIA DETECTADA\n\n"
        "Com base na sua descricao, esta pode ser uma situacao de urgencia medica.\n"
        "Tome as seguintes acoes IMEDIATAMENTE:\n\n"
        "1. Ligue para o SAMU: 192\n"
        "2. Ligue para os Bombeiros: 193\n"
        "3. Va ao pronto-socorro mais proximo\n\n"
        "Nao espere - em emergencias, cada segundo conta."
    )
    state["sources"] = ["Protocolo de Emergencias Medicas"]
    state["log"].append("[EMERGENCIA] Resposta de emergencia gerada.")
    return state


def node_busca_rag(state: MedState) -> MedState:
    """Busca documentos relevantes no vector store FAISS."""
    try:
        from langchain_community.vectorstores import FAISS
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        if not VECTOR_STORE_PATH.exists():
            state["log"].append("[RAG] Vector store nao encontrado. Execute chains.py primeiro.")
            state["retrieved_docs"] = []
            state["sources"] = []
            return state

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vs = FAISS.load_local(
            str(VECTOR_STORE_PATH), embeddings,
            allow_dangerous_deserialization=True,
        )
        docs = vs.similarity_search(state["question"], k=TOP_K_RETRIEVAL)
        state["retrieved_docs"] = [d.page_content for d in docs]
        state["sources"] = [d.metadata.get("question", "")[:100] for d in docs]
        state["log"].append(f"[RAG] {len(docs)} documentos recuperados.")
    except Exception as e:
        state["log"].append(f"[RAG] Erro: {e}")
        state["retrieved_docs"] = []
        state["sources"] = []
    return state


def node_geracao(state: MedState) -> MedState:
    """Gera resposta via backend configurado (OpenAI ou Phi-3 local) com contexto RAG."""
    if state["category"] == "fora_escopo":
        state["raw_answer"] = (
            "Ola! Sou o MedAssist, assistente de informacoes medicas baseado no MedQuAD. "
            "Posso ajuda-lo com duvidas sobre sintomas, doencas, tratamentos e prevencao. "
            "Como posso ajudar?"
        )
        state["log"].append("[GERACAO] Saudacao gerada.")
        return state

    try:
        context = "\n\n".join(state["retrieved_docs"][:3]) if state["retrieved_docs"] else ""
        answer  = llm_generate(question=state["question"], context=context)
        state["raw_answer"] = answer
        state["log"].append(
            f"[GERACAO:{LLM_BACKEND.upper()}] Resposta gerada ({len(answer)} chars)."
        )
    except Exception as e:
        state["raw_answer"] = f"Erro ao gerar resposta: {e}"
        state["log"].append(f"[GERACAO] Erro: {e}")

    return state


def node_validacao(state: MedState) -> MedState:
    """Valida resposta, adiciona flags de seguranca, fontes e disclaimer."""
    answer = state["raw_answer"]
    flags = []

    if any(kw in answer.lower() for kw in PRESCRIPTION_KEYWORDS):
        flags.append("AVISO: Resposta menciona medicacao. Revisao humana recomendada.")

    state["flags"].extend(flags)

    sources_block = ""
    if state["sources"]:
        src_list = "\n".join(f"  - {s}" for s in state["sources"][:3])
        sources_block = f"\n\nFontes consultadas (MedQuAD):\n{src_list}"

    flags_block = ""
    if state["flags"]:
        flags_block = "\n\nAvisos:\n" + "\n".join(f"  - {f}" for f in state["flags"])

    state["final_answer"] = answer + sources_block + flags_block + SAFETY_DISCLAIMER
    state["log"].append(f"[VALIDACAO] Resposta final montada. Flags: {state['flags']}")
    return state


# ─────────────────────────────────────────────
# Roteamento
# ─────────────────────────────────────────────

def route_after_triagem(state: MedState) -> str:
    return "node_emergencia" if state["category"] == "emergencia" else "node_busca_rag"


# ─────────────────────────────────────────────
# Build e execucao do grafo
# ─────────────────────────────────────────────

def build_graph():
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        print("ERRO: pip install langgraph")
        sys.exit(1)

    graph = StateGraph(MedState)
    graph.add_node("node_entrada",    node_entrada)
    graph.add_node("node_triagem",    node_triagem)
    graph.add_node("node_emergencia", node_emergencia)
    graph.add_node("node_busca_rag",  node_busca_rag)
    graph.add_node("node_geracao",    node_geracao)
    graph.add_node("node_validacao",  node_validacao)

    graph.set_entry_point("node_entrada")
    graph.add_edge("node_entrada",    "node_triagem")
    graph.add_conditional_edges("node_triagem", route_after_triagem)
    graph.add_edge("node_emergencia", "node_validacao")
    graph.add_edge("node_busca_rag",  "node_geracao")
    graph.add_edge("node_geracao",    "node_validacao")
    graph.add_edge("node_validacao",  END)

    return graph.compile()


def run_graph(question: str) -> dict:
    app = build_graph()
    initial: MedState = {
        "question": question,
        "category": None,
        "retrieved_docs": [],
        "raw_answer": "",
        "final_answer": "",
        "sources": [],
        "flags": [],
        "log": [],
    }
    return app.invoke(initial)


def main():
    sep = "=" * 60
    info = get_backend_info()
    print(f"\n{sep}")
    print(f"  FASE 3 - Fluxo LangGraph")
    print(f"  Backend: {info['description']}")
    print(sep)

    if LLM_BACKEND == "openai" and not OPENAI_API_KEY:
        print("ERRO: LLM_BACKEND=openai mas OPENAI_API_KEY nao configurada no .env")
        print("   Ou mude para: LLM_BACKEND=local")
        sys.exit(1)

    test_cases = [
        ("Consulta normal",    "Quais sao os sintomas do diabetes?"),
        ("Emergencia",         "Meu pai esta com dor no peito intensa e nao respira bem"),
        ("Fora do escopo",     "Ola, tudo bem?"),
        ("Possivel prescricao","Que remedio devo tomar para pressao alta?"),
    ]

    for label, question in test_cases:
        print(f"\n[{label.upper()}]")
        print(f"Pergunta: {question}")
        result = run_graph(question)
        print(f"Categoria: {result['category']}")
        print(f"Flags:     {result['flags']}")
        print(f"Resposta:  {result['final_answer'][:300]}...")
        print("Log:")
        for entry in result["log"]:
            print(f"   {entry}")
        print("-" * 40)

    diagram_path = _root / "outputs" / "langgraph_diagram.txt"
    diagram_path.parent.mkdir(exist_ok=True)
    with open(diagram_path, "w", encoding="utf-8") as f:
        f.write(
            "DIAGRAMA DO FLUXO LANGGRAPH - MedAssist\n"
            "========================================\n\n"
            "[ENTRADA]\n"
            "    |\n"
            "    v\n"
            "[TRIAGEM] --- emergencia ---> [EMERGENCIA] ---+\n"
            "    |                                         |\n"
            "    v                                         |\n"
            "[BUSCA RAG]                                   |\n"
            "    |                                         |\n"
            "    v                                         |\n"
            "[GERACAO via OpenAI]                          |\n"
            "    |                                         |\n"
            "    +--------------------+--------------------+\n"
            "                         |\n"
            "                         v\n"
            "                    [VALIDACAO]\n"
            "                         |\n"
            "                         v\n"
            "                       [FIM]\n"
        )
    print(f"\nDiagrama salvo em: {diagram_path}")
    print(f"\n{sep}")
    print("  FASE 3.2 - Fluxo LangGraph testado com sucesso!")
    print(sep)


if __name__ == "__main__":
    main()
