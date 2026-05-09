"""
FASE 4 — safety_guard.py
Guardrails e limites de atuação do assistente médico.

Execute: python fase4_seguranca/safety_guard.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from utils import fix_windows_encoding
fix_windows_encoding()

import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EMERGENCY_KEYWORDS, PRESCRIPTION_KEYWORDS, SAFETY_DISCLAIMER


class SafetyGuard:
    """
    Filtra e valida entradas/saídas do assistente médico.
    Garante que o assistente não presceva medicamentos diretamente
    e detecta emergências para redirecionamento adequado.
    """

    OUT_OF_SCOPE_PATTERNS = [
        r"\b(receita|prescreva|prescrevas|prescreva-me)\b",
        r"\btome\s+\d+\s*mg\b",
        r"\busem?\s+(o\s+remédio|a\s+medicação)\b",
    ]

    SENSITIVE_TOPICS = [
        "suicídio", "automutilação", "me matar", "quero morrer",
        "overdose intencional",
    ]

    def __init__(self):
        self._out_of_scope_re = [re.compile(p, re.IGNORECASE) for p in self.OUT_OF_SCOPE_PATTERNS]

    def check_input(self, question: str) -> dict:
        """
        Analisa a pergunta do usuário.
        Retorna dict com:
          - allow: bool — se a pergunta deve ser processada
          - category: str — tipo de detecção
          - message: str — mensagem de override (se houver)
        """
        q = question.lower()

        # Emergências
        if any(kw in q for kw in EMERGENCY_KEYWORDS):
            return {
                "allow": False,
                "category": "emergencia",
                "message": (
                    "🚨 **EMERGÊNCIA DETECTADA**\n\n"
                    "Ligue imediatamente:\n"
                    "• SAMU: **192**\n"
                    "• Bombeiros: **193**\n"
                    "• Pronto-socorro mais próximo\n\n"
                    "Não perca tempo — esta é uma situação de urgência médica."
                ),
            }

        # Tópicos sensíveis (saúde mental)
        if any(kw in q for kw in self.SENSITIVE_TOPICS):
            return {
                "allow": False,
                "category": "saude_mental_critica",
                "message": (
                    "💙 Percebi que você pode estar passando por um momento difícil.\n\n"
                    "Por favor, entre em contato com o CVV — Centro de Valorização da Vida:\n"
                    "• Telefone: **188** (24 horas, gratuito)\n"
                    "• Chat: **cvv.org.br**\n\n"
                    "Você não está sozinho(a). Há pessoas prontas para ouvir."
                ),
            }

        return {"allow": True, "category": "consulta", "message": ""}

    def check_output(self, answer: str) -> dict:
        """
        Analisa a resposta gerada pelo LLM.
        Retorna dict com:
          - safe: bool
          - warnings: list[str]
          - sanitized_answer: str
        """
        warnings = []

        # Verifica prescrição direta
        for pattern in self._out_of_scope_re:
            if pattern.search(answer):
                warnings.append(
                    "⚠️ AVISO: Esta resposta pode conter sugestão de medicação. "
                    "A decisão de prescrever é exclusiva do médico."
                )
                break

        # Verifica menção de dosagem
        if re.search(r"\d+\s*mg", answer, re.IGNORECASE):
            warnings.append(
                "⚠️ AVISO: Dosagens de medicamentos mencionadas. "
                "Sempre consulte um médico antes de usar qualquer medicação."
            )

        return {
            "safe": len(warnings) == 0,
            "warnings": warnings,
            "sanitized_answer": answer,
        }

    def apply(self, question: str, answer: str) -> str:
        """Aplica todos os guardrails e retorna resposta final segura."""
        input_check = self.check_input(question)
        if not input_check["allow"]:
            return input_check["message"] + SAFETY_DISCLAIMER

        output_check = self.check_output(answer)
        final = output_check["sanitized_answer"]

        if output_check["warnings"]:
            warnings_block = "\n".join(output_check["warnings"])
            final = f"{final}\n\n{warnings_block}"

        return final + SAFETY_DISCLAIMER


def demo():
    sep = "=" * 60
    print(f"\n{sep}")
    print("  🔒 FASE 4 — Safety Guard")
    print(sep)

    guard = SafetyGuard()

    test_cases = [
        # (tipo, pergunta, resposta_simulada)
        ("Normal", "Quais são os sintomas da diabetes?",
         "Os sintomas incluem sede excessiva, urina frequente e cansaço."),
        ("Emergência", "Meu filho está inconsciente e não respira!",
         "Não deveria chegar aqui."),
        ("Prescrição direta", "Que remédio para pressão?",
         "Tome 25mg de losartana por dia."),
        ("Saúde mental", "Estou querendo me matar, não consigo mais",
         "Não deveria chegar aqui."),
        ("Dosagem", "Como tomar o paracetamol?",
         "O paracetamol é usado em doses de 500mg a 1000mg."),
    ]

    results = []
    for tipo, question, answer in test_cases:
        print(f"\n[{tipo.upper()}]")
        print(f"❓ Pergunta: {question}")

        input_check = guard.check_input(question)
        if not input_check["allow"]:
            final = guard.apply(question, answer)
            print(f"🚫 Bloqueado ({input_check['category']})")
            print(f"📢 Resposta: {final[:200]}...")
            results.append(("BLOQUEADO", tipo))
        else:
            output_check = guard.check_output(answer)
            final = guard.apply(question, answer)
            print(f"✅ Permitido | Warnings: {output_check['warnings']}")
            print(f"💬 Resposta: {final[:200]}...")
            results.append(("PERMITIDO" if output_check["safe"] else "AVISO", tipo))

    print(f"\n📊 Resumo:")
    for status, tipo in results:
        icon = "✅" if status == "PERMITIDO" else ("🚫" if status == "BLOQUEADO" else "⚠️")
        print(f"   {icon} [{status}] {tipo}")

    print(f"\n{sep}")
    print("  ✅ Fase 4.1 — Safety Guard validado com sucesso!")
    print(sep)


if __name__ == "__main__":
    demo()
