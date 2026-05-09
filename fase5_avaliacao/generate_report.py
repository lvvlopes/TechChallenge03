"""
FASE 5 — generate_report.py
Gera o relatório técnico final do projeto MedAssist.

Execute: python fase5_avaliacao/generate_report.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from utils import fix_windows_encoding
fix_windows_encoding()

import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

_root = Path(__file__).parent.parent
REPORT_PATH = _root / "outputs" / "relatorio_tecnico_final.md"


def load_json_safe(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_text_safe(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "(arquivo não encontrado — execute as fases anteriores)"


def generate_report() -> str:
    benchmark = load_json_safe(_root / "outputs" / "benchmark_results.json")
    eval_report = load_text_safe(_root / "outputs" / "evaluation_report.txt")
    data_quality = load_text_safe(_root / "outputs" / "data_quality_report.txt")

    summary = benchmark.get("summary", {})
    ts = datetime.now().strftime("%d/%m/%Y %H:%M")

    report = f"""# 📋 Relatório Técnico Final — MedAssist
### Tech Challenge IADT — Fase 3 | POSTECH / FIAP
**Gerado em:** {ts}

---

## 1. Visão Geral do Projeto

O **MedAssist** é um assistente virtual médico desenvolvido como entregável do Tech Challenge Fase 3. O sistema integra fine-tuning de LLM com dados médicos (MedQuAD PT-BR) e orquestração de fluxos de decisão via LangChain e LangGraph.

**Objetivo:** Criar um assistente capaz de auxiliar médicos e profissionais de saúde em condutas clínicas, respondendo dúvidas baseadas em protocolos médicos e sugerindo procedimentos com base em evidências — sempre com validação humana obrigatória.

---

## 2. Dataset Utilizado

| Atributo | Valor |
|---------|-------|
| **Nome** | MedQuAD PT-BR |
| **Total de amostras** | 5.274 pares pergunta/resposta |
| **Idioma** | Português Brasileiro |
| **Formato** | JSONL com roles: system / user / assistant |
| **Conteúdo** | Perguntas e respostas clínicas sobre doenças, sintomas, tratamentos e prevenção |

### 2.1 Relatório de Qualidade dos Dados

```
{data_quality[:800]}
```

---

## 3. Processo de Fine-tuning

### 3.1 Modelo Base
O sistema foi desenvolvido para operar com modelos LLM de diversas escalas:
- **Com GPU:** LLaMA 3 / Mistral 7B com QLoRA (4-bit quantization)
- **Sem GPU:** GPT-2 como demonstração do pipeline

### 3.2 Técnica: LoRA / QLoRA

O fine-tuning utiliza **LoRA (Low-Rank Adaptation)** via biblioteca PEFT, que treina apenas adaptadores de baixo rank ao invés de todos os parâmetros do modelo:

```
Parâmetros LoRA configurados:
- rank (r): 16
- alpha: 32
- dropout: 0.05
- quantização: 4-bit NF4 (QLoRA)
```

### 3.3 Parâmetros de Treinamento

| Parâmetro | Valor |
|-----------|-------|
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation | 4 steps |
| Learning rate | 2e-4 |
| Warmup ratio | 0.03 |
| Max seq length | 512 tokens |

### 3.4 Resultados do Fine-tuning

```
{eval_report[:600] if eval_report else "(Execute python fase2_finetuning/evaluate_model.py)"}
```

---

## 4. Arquitetura do Assistente Médico

### 4.1 Pipeline RAG (LangChain)

O assistente utiliza **Retrieval-Augmented Generation (RAG)** para fundamentar suas respostas no dataset MedQuAD:

```
[Pergunta do Usuário]
       │
       ▼
[Embeddings] ← sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
       │
       ▼
[FAISS Vector Store] ← top-3 documentos mais similares
       │
       ▼
[Prompt Template] ← contexto + pergunta
       │
       ▼
[LLM Fine-tuned] → resposta fundamentada
       │
       ▼
[Resposta + Fontes + Disclaimer]
```

### 4.2 Fluxo LangGraph (Decisão Automatizada)

```
[ENTRADA]
    │
    ▼
[TRIAGEM] ─── detecta emergência? ──► [ALERTA EMERGÊNCIA]
    │                                         │
    │ não                                     │
    ▼                                         │
[BUSCA RAG]                                  │
    │                                         │
    ▼                                         │
[GERAÇÃO]                                    │
    │                                         │
    └──────────────────────┬──────────────────┘
                           ▼
                      [VALIDAÇÃO]
                      (safety + disclaimer + fontes)
                           │
                           ▼
                        [SAÍDA]
```

**Nós implementados:**
- `node_entrada`: normalização da pergunta
- `node_triagem`: classificação (emergência / consulta / fora_escopo)
- `node_emergencia`: resposta com redirecionamento para SAMU/Bombeiros
- `node_busca_rag`: recuperação de documentos relevantes
- `node_geracao`: síntese da resposta
- `node_validacao`: aplicação de guardrails e disclaimers

---

## 5. Segurança e Validação

### 5.1 Guardrails Implementados

| Tipo | Descrição | Ação |
|------|-----------|------|
| Emergência | Detecta palavras como "não respira", "inconsciente", "infarto" | Bloqueia e redireciona para SAMU |
| Prescrição direta | Detecta padrões de prescrição em respostas | Adiciona aviso de revisão humana |
| Saúde mental crítica | Detecta indicativos de automutilação | Redireciona para CVV (188) |
| Dosagem | Detecta menção a dosagens em mg | Adiciona aviso de consulta médica |

### 5.2 Logging e Auditoria

Todas as interações são registradas em `logs/audit.log` com:
- Timestamp UTC
- Session ID
- Categoria da interação
- Preview da pergunta e resposta
- Fontes utilizadas
- Flags de segurança acionadas

### 5.3 Explainability

Cada resposta inclui:
- Score de confiança (0–100%)
- Método de geração (RAG ou LLM puro)
- Fontes utilizadas (documentos recuperados)
- Disclaimer médico obrigatório

---

## 6. Benchmark e Avaliação

### 6.1 Resultados do Benchmark

| Métrica | Valor |
|---------|-------|
| Taxa de sucesso (QA) | {summary.get('qa_success_rate', 'N/A')} |
| Latência média | {summary.get('avg_latency_s', 'N/A')}s |
| Taxa de aprovação (segurança) | {summary.get('safety_pass_rate', 'N/A')} |

### 6.2 Análise Qualitativa

- **Pontos fortes:** Respostas fundamentadas no dataset médico; disclaimer sempre presente; detecção de emergências funcional
- **Limitações:** Modelo base GPT-2 (demonstração) tem qualidade de geração limitada; modelos maiores (LLaMA, Mistral) necessitam GPU
- **Melhorias futuras:** Fine-tuning com mais epochs, avaliação humana das respostas, expansão do dataset

---

## 7. Conclusão

O MedAssist demonstra a viabilidade de construir um assistente médico em português com:
- Fine-tuning eficiente via LoRA/QLoRA
- RAG para fundamentação em evidências
- Fluxos de decisão automatizados com LangGraph
- Guardrails robustos para segurança clínica

O sistema está estruturado de forma modular e extensível, permitindo evolução incremental conforme novos dados e modelos estejam disponíveis.

---

*Relatório gerado automaticamente por `fase5_avaliacao/generate_report.py`*
"""
    return report


def main():
    sep = "=" * 60
    print(f"\n{sep}")
    print("  📝 FASE 5 — Geração do Relatório Técnico Final")
    print(sep)

    report = generate_report()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Relatório técnico gerado com sucesso!")
    print(f"📄 Localização: {REPORT_PATH}")
    print(f"📏 Tamanho: {len(report):,} caracteres")

    print(f"\n{sep}")
    print("  ✅ Fase 5 — Relatório Final concluído!")
    print(sep)


if __name__ == "__main__":
    main()
