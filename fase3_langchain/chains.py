"""
FASE 3 - chains.py
Pipeline RAG com LangChain: embeddings do MedQuAD + OpenAI como LLM.
Usa LCEL (LangChain Expression Language) — compativel com LangChain >= 0.2.

Execute: python fase3_langchain/chains.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import fix_windows_encoding
fix_windows_encoding()

from config import (
    DATASET_PATH, VECTOR_STORE_PATH, EMBEDDING_MODEL,
    TOP_K_RETRIEVAL, SAFETY_DISCLAIMER,
    OPENAI_API_KEY, OPENAI_MODEL, LLM_BACKEND,
)
from fase3_langchain.llm_backend import generate as llm_generate, get_backend_info

_root = Path(__file__).parent.parent


def check_openai_key():
    if LLM_BACKEND == "openai" and not OPENAI_API_KEY:
        print("ERRO: LLM_BACKEND=openai mas OPENAI_API_KEY nao encontrada.")
        print("   Edite o .env e preencha: OPENAI_API_KEY=sk-...")
        print("   Ou mude para: LLM_BACKEND=local")
        sys.exit(1)


def get_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_documents() -> list[dict]:
    candidates = [DATASET_PATH, _root / "data" / "medquad_ptbr.jsonl"]
    path = next((p for p in candidates if Path(p).exists()), None)
    if not path:
        print("ERRO: medquad_ptbr.jsonl nao encontrado em data/")
        sys.exit(1)

    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                msgs = r.get("messages", [])
                q = next((m["content"] for m in msgs if m["role"] == "user"), "")
                a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                if q and a:
                    docs.append({"question": q, "answer": a,
                                 "text": f"Pergunta: {q}\nResposta: {a}"})
    return docs


def build_vector_store(docs: list[dict], force_rebuild: bool = False):
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    embeddings  = get_embeddings()
    index_path  = str(VECTOR_STORE_PATH)
    faiss_file  = VECTOR_STORE_PATH / "index.faiss"

    if not force_rebuild and faiss_file.exists():
        print("Carregando vector store existente...")
        return FAISS.load_local(index_path, embeddings,
                                allow_dangerous_deserialization=True)

    print(f"Construindo vector store com {len(docs)} documentos...")
    print(f"Modelo de embeddings: {EMBEDDING_MODEL}")
    print("(primeira execucao pode levar alguns minutos)")

    lc_docs = [Document(page_content=d["text"],
                        metadata={"question": d["question"]}) for d in docs]

    vectorstore = FAISS.from_documents(lc_docs, embeddings)
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(index_path)
    print(f"Vector store salvo em: {index_path}")
    return vectorstore


def build_rag_chain(vectorstore):
    """Chain RAG usando llm_backend — suporta OpenAI ou Phi-3-mini local."""

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_invoke(inputs: dict) -> str:
        question = inputs["question"]
        docs     = retriever.invoke(question)
        context  = format_docs(docs)
        return llm_generate(question=question, context=context)

    return rag_invoke, retriever


def query_rag(chain_tuple, question: str) -> dict:
    chain, retriever = chain_tuple
    # chain é uma função Python pura — chamar diretamente
    answer       = chain({"question": question})
    source_docs  = retriever.invoke(question)
    source_texts = [doc.metadata.get("question", "")[:100] for doc in source_docs]

    return {
        "question": question,
        "answer": answer + SAFETY_DISCLAIMER,
        "sources": source_texts,
        "num_sources": len(source_docs),
    }


def demo():
    sep = "=" * 60
    info = get_backend_info()
    print(f"\n{sep}")
    print(f"  FASE 3 - Pipeline RAG")
    print(f"  Backend: {info['description']}")
    print(sep)

    check_openai_key()

    print("\nCarregando documentos MedQuAD...")
    docs = load_documents()
    print(f"   {len(docs)} documentos carregados.")

    vectorstore   = build_vector_store(docs)
    print("Construindo chain RAG...")
    chain_tuple   = build_rag_chain(vectorstore)

    test_questions = [
        "Quais sao os sintomas do diabetes tipo 2?",
        "Como se previne a hipertensao?",
        "O que e leucemia?",
    ]

    print(f"\nTestando {len(test_questions)} perguntas:\n")
    for q in test_questions:
        print(f"Pergunta: {q}")
        result = query_rag(chain_tuple, q)
        print(f"Resposta: {result['answer'][:350]}...")
        print(f"Fontes ({result['num_sources']}): {result['sources'][:2]}")
        print("-" * 50)

    print(f"\n{sep}")
    print("  FASE 3.1 - Pipeline RAG testado com sucesso!")
    print(sep)

    return chain_tuple


if __name__ == "__main__":
    demo()
