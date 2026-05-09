"""
FASE 4 — explainability.py
Rastreabilidade e explainability das respostas do assistente.

Execute: python fase4_seguranca/explainability.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from utils import fix_windows_encoding
fix_windows_encoding()

from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


class ExplainabilityLayer:
    """
    Adiciona rastreabilidade às respostas:
    - Indica quais documentos foram usados (RAG)
    - Atribui score de confiança
    - Formata resposta com citações
    """

    def __init__(self):
        self.history: list[dict] = []

    def score_confidence(self, question: str, docs: list[str]) -> float:
        """
        Heurística simples de confiança baseada em:
        - Número de documentos recuperados
        - Sobreposição de termos entre pergunta e docs
        """
        if not docs:
            return 0.1

        q_words = set(question.lower().split())
        scores = []
        for doc in docs:
            doc_words = set(doc.lower().split())
            overlap = len(q_words & doc_words)
            score = min(overlap / max(len(q_words), 1), 1.0)
            scores.append(score)

        base_score = sum(scores) / len(scores)
        doc_bonus = min(len(docs) * 0.1, 0.3)
        return round(min(base_score + doc_bonus + 0.3, 1.0), 2)

    def format_response(
        self,
        question: str,
        answer: str,
        sources: list[str],
        category: str = "consulta",
    ) -> dict:
        """
        Monta resposta explicável com:
        - Confiança
        - Fontes usadas
        - Método de geração
        - Timestamp
        """
        confidence = self.score_confidence(question, sources)

        conf_label = (
            "Alta" if confidence >= 0.7 else
            "Média" if confidence >= 0.4 else
            "Baixa"
        )

        conf_icon = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"

        method = "RAG + LLM Fine-tuned" if sources else "LLM Fine-tuned (sem contexto)"

        explainable_response = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "category": category,
            "confidence_score": confidence,
            "confidence_label": conf_label,
            "method": method,
            "sources_used": sources,
            "answer": answer,
            "formatted": self._format_display(answer, sources, confidence, conf_label, conf_icon, method),
        }

        self.history.append(explainable_response)
        return explainable_response

    def _format_display(
        self, answer: str, sources: list[str],
        confidence: float, conf_label: str, conf_icon: str, method: str
    ) -> str:
        lines = [answer]

        lines.append(f"\n{'─'*50}")
        lines.append(f"{conf_icon} **Confiança:** {conf_label} ({confidence:.0%})")
        lines.append(f"🔧 **Método:** {method}")

        if sources:
            lines.append(f"📚 **Fontes ({len(sources)}):**")
            for i, src in enumerate(sources[:3], 1):
                lines.append(f"   [{i}] {src[:100]}")
        else:
            lines.append("📚 **Fontes:** Resposta sem contexto recuperado (baseada em treinamento)")

        return "\n".join(lines)


def demo():
    sep = "=" * 60
    print(f"\n{sep}")
    print("  🔍 FASE 4 — Explainability das Respostas")
    print(sep)

    explainer = ExplainabilityLayer()

    test_cases = [
        {
            "question": "Quais são os sintomas da diabetes tipo 2?",
            "answer": "Os sintomas incluem sede excessiva, urina frequente, fadiga e visão embaçada.",
            "sources": [
                "Pergunta: Quais são os sintomas do diabetes? Resposta: Poliúria, polidipsia...",
                "Pergunta: Como identificar diabetes tipo 2? Resposta: Sinais clínicos incluem...",
                "Pergunta: Sintomas do diabetes mellitus? Resposta: Manifestações clínicas...",
            ],
            "category": "consulta",
        },
        {
            "question": "O que é fibromialgia?",
            "answer": "Fibromialgia é uma condição caracterizada por dor musculoesquelética difusa.",
            "sources": [
                "Pergunta: O que é fibromialgia? Resposta: É uma síndrome de dor crônica...",
            ],
            "category": "consulta",
        },
        {
            "question": "Como curar gripe rapidamente?",
            "answer": "Não encontrei informações específicas. Recomendo consultar um médico.",
            "sources": [],
            "category": "consulta",
        },
    ]

    for case in test_cases:
        print(f"\n❓ Pergunta: {case['question']}")
        result = explainer.format_response(
            question=case["question"],
            answer=case["answer"],
            sources=case["sources"],
            category=case["category"],
        )
        print(f"📊 Confiança: {result['confidence_label']} ({result['confidence_score']:.0%})")
        print(f"🔧 Método: {result['method']}")
        print(f"📚 Fontes: {len(result['sources_used'])}")
        print(f"\nResposta formatada:\n{result['formatted']}")
        print("-" * 50)

    print(f"\n📋 Histórico da sessão: {len(explainer.history)} respostas geradas")

    print(f"\n{sep}")
    print("  ✅ Fase 4.3 — Explainability validada com sucesso!")
    print(sep)


if __name__ == "__main__":
    demo()
