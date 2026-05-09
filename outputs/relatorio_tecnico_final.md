# Relatório Técnico Final — MedAssist
## Tech Challenge Fase 3 · POSTECH / FIAP
**Data:** Maio de 2026

---

## 1. Resumo Executivo

O **MedAssist** é um assistente virtual médico desenvolvido para o Tech Challenge Fase 3. Integra fine-tuning real de LLM com dados médicos em português, orquestração de fluxos com LangChain e LangGraph, sistema de segurança clínica com guardrails, interface web profissional e três backends de inferência — incluindo deploy em nuvem Azure.

**Resultado:** 15/16 requisitos obrigatórios atendidos. Adaptadores LoRA validados (28/28). Interface web funcional.

---

## 2. Arquitetura

```
[Interface Web — http://localhost:8080]
         │
  [web_app.py — FastAPI]
         │
 [LangGraph — Orquestrador]
    │              │
[Triagem]    [Safety Guard]
    │
 ┌──┴──────────┐
 │             │
[Emergência] [RAG FAISS]
             [MedQuAD PT-BR]
                  │
          [llm_backend.py]
       ┌─────────┬─────────┐
  [OpenAI]  [Local]   [Azure]
  API        GPU       Nuvem
                  │
   [Resposta + Fonte + Disclaimer]
                  │
          [Logging / Audit]
```

| Componente | Tecnologia | Arquivo |
|---|---|---|
| Fine-tuning | Phi-3-mini-4k + QLoRA via Unsloth | `MedAssist_FineTuning_Colab.ipynb` |
| Adaptadores LoRA | 114 MB · 28/28 OK | `outputs/model/` |
| Backend A | OpenAI gpt-4o-mini | `.env` — MODO 1 |
| Backend B | Phi-3+LoRA via local_server.py | `local_server.py` — MODO 2 |
| Backend C | Azure Container Apps | `Dockerfile` + `AZURE_DEPLOY.md` — MODO 3 |
| RAG | FAISS + MiniLM-L12 | `chains.py` |
| Orquestração | LangChain + LangGraph | `graph.py` |
| Segurança | Safety Guard + Logging + Explainability | `fase4_seguranca/` |
| Interface | FastAPI + HTML/CSS/JS | `web_app.py` + `web_ui.html` |

---

## 3. Fine-tuning de LLM

### 3.1 Processo de Treinamento (Google Colab T4)

| Parâmetro | Valor |
|---|---|
| Modelo base | Phi-3-mini-4k-instruct (3.8B parâmetros) |
| Técnica | QLoRA 4-bit (NF4) via Unsloth |
| LoRA rank / alpha | r=16 / α=32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Dataset treino | 4.747 amostras MedQuAD PT-BR |
| Epochs | 1 (~45 min T4) · Batch efetivo: 16 |
| Learning rate | 2e-4 |

### 3.2 Validação dos Adaptadores — 28/28 OK

```
Arquivos:   adapter_config.json, adapter_model.safetensors (114.1 MB),
            tokenizer.json (32k vocab), tokenizer.model, chat_template.jinja
Config:     peft_type=LORA, task_type=CAUSAL_LM, rank=16, alpha=32,
            inference_mode=True
Pesos:      448 tensores · 224 pares lora_A/lora_B · formato PyTorch
```

### 3.3 Avaliação Qualitativa (5/5 aprovadas no Colab)

| Pergunta | Resultado |
|---|---|
| Sintomas do diabetes tipo 2 | 8 sintomas + orientação |
| Causas da hipertensão | 8 fatores causais |
| Como funciona a quimioterapia | Mecanismo, vias, efeitos colaterais |
| Sintomas de AVC | 6 sinais + urgência imediata |
| O que é leucemia linfoblástica aguda | Definição clínica completa |

### 3.4 Decisão Arquitetural sobre Inferência

O Phi-3-mini requer ~8 GB VRAM. Sem GPU local disponível, adotou-se:
- **Fine-tuning (Colab):** comprova o treinamento real — adaptadores em `outputs/model/`
- **Inferência:** backend configurável — OpenAI, local (GPU), ou Azure Container Apps
- **Alternância em tempo real:** modal da interface web troca o backend sem reiniciar

---

## 4. Assistente Médico com LangChain + LangGraph

### 4.1 Pipeline RAG

Recuperação semântica com FAISS sobre 5.274 documentos MedQuAD PT-BR usando embeddings multilinguais (MiniLM-L12-v2). Top-3 documentos injetados no prompt como contexto.

### 4.2 Fluxo LangGraph — 6 nós

```
[ENTRADA] → [TRIAGEM] → [EMERGÊNCIA] (SAMU 192 / CVV 188)
                     ↘ [BUSCA RAG] → [GERAÇÃO] → [VALIDAÇÃO] → [SAÍDA]
```

Roteamento condicional: após triagem, o grafo decide automaticamente entre emergência e consulta.

### 4.3 Três Backends de Inferência

| Modo | Configuração | Uso |
|---|---|---|
| OpenAI API | `OPENAI_API_KEY=sk-...` | Desenvolvimento, demo rápida |
| Servidor Local | `OPENAI_BASE_URL=http://localhost:8000/v1` | Demo Phi-3 com GPU |
| Azure | `OPENAI_BASE_URL=https://<app>.azurecontainerapps.io/v1` | Acesso externo |

---

## 5. Segurança e Validação

### 5.1 Guardrails

| Categoria | Gatilho | Ação |
|---|---|---|
| Emergência | "infarto", "AVC", "parada cardíaca"... | Bloqueia → SAMU 192 |
| Saúde mental | "suicídio", "me matar"... | Redireciona → CVV 188 |
| Prescrição | Padrões de prescrição na resposta | Aviso revisão humana |
| Dosagem | mg/ml na resposta | Aviso consulta médica |
| Fora do escopo | Saudações, perguntas não médicas | Categoria `fora_escopo` |

### 5.2 Logging (logs/audit.log — JSON)

```json
{
  "timestamp": "2026-04-30T03:45:34Z",
  "session_id": "20260430_004106",
  "category": "consulta",
  "sources": ["O que é Miocardite?"],
  "flags": [],
  "has_disclaimer": true
}
```

### 5.3 Explainability

Cada resposta inclui: fontes MedQuAD utilizadas, score de confiança (0–100%), método de geração identificado, disclaimer médico obrigatório.

---

## 6. Deploy Azure Container Apps

Arquivos entregues: `Dockerfile` + `AZURE_DEPLOY.md`

Fluxo resumido:
```bash
docker build -t medassist-phi3 .
docker push <registry>.azurecr.io/medassist-phi3:latest
az containerapp create --name medassist-phi3 ... --ingress external
# → URL pública HTTPS gerada automaticamente
```

Custo estimado: ~$0.50–1.50/h. Scale-to-zero disponível (sem custo quando inativo).

---

## 7. Conformidade com a Especificação TC F3

### Requisitos Obrigatórios

| # | Requisito | Status | Evidência |
|---|---|---|---|
| 1 | Fine-tuning de LLM com dados médicos | ✅ | Phi-3-mini QLoRA, Colab T4, 28/28 OK |
| 2 | Preprocessing, anonimização, curadoria | ✅ | `fase1_dados/preprocess.py` |
| 3 | Pipeline LangChain integrando LLM | ✅ | `chains.py` — RAG FAISS |
| 4 | Consultas em base de dados estruturada | ✅ | FAISS 5.274 docs MedQuAD |
| 5 | Contextualização com dados do paciente | ✅ | Top-3 docs RAG no prompt |
| 6 | Fluxos LangGraph automatizados | ✅ | `graph.py` — 6 nós |
| 7 | Limites de atuação (sem prescrição direta) | ✅ | `safety_guard.py` — 5 categorias |
| 8 | Logging para rastreamento e auditoria | ✅ | `logs/audit.log` JSON |
| 9 | Explainability (fontes na resposta) | ✅ | `explainability.py` + fontes UI |
| 10 | Projeto modularizado em Python | ✅ | 5 fases + `__init__.py` |
| 11 | Instruções no README | ✅ | `README.md` completo |
| 12 | Pipeline fine-tuning no repositório | ✅ | `MedAssist_FineTuning_Colab.ipynb` |
| 13 | Integração LangChain | ✅ | `fase3_langchain/` completo |
| 14 | Fluxos LangGraph | ✅ | `graph.py` + `outputs/langgraph_diagram.txt` |
| 15 | Dataset anonimizado | ✅ | `data/medquad_ptbr.jsonl` |
| 16 | Relatório técnico detalhado | ✅ | Este documento |

**Pendente:** Vídeo demonstrativo (até 15 min) — a produzir para entrega.

### Entregáveis Adicionais

| Entregável | Descrição |
|---|---|
| Interface web profissional | `web_app.py` + `web_ui.html` — sistema completo no browser |
| 3 backends de inferência | OpenAI, Servidor Local, Azure — alternáveis em tempo real |
| Deploy Azure | `Dockerfile` + `AZURE_DEPLOY.md` — guia completo com CLI |
| Servidor local | `local_server.py` — API OpenAI-compatible para Phi-3-mini |

---

## 8. Roteiro para Demonstração em Vídeo (até 15 min)

1. **(2 min)** Contextualização: problema, solução, arquitetura
2. **(3 min)** Interface web: perguntas rápidas, chat, fontes MedQuAD, badges
3. **(2 min)** Demonstrar emergência: bolha vermelha + SAMU 192
4. **(2 min)** Trocar backend pelo modal: Azure Container Apps
5. **(2 min)** Painel de detalhes: categoria, tempo, fontes, log
6. **(2 min)** Fine-tuning: notebook Colab, adaptadores, validação 28/28
7. **(1 min)** Conclusão

---

*MedAssist · Tech Challenge Fase 3 · POSTECH / FIAP · Maio 2026*
