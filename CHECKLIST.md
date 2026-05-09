# CHECKLIST — MedAssist Tech Challenge Fase 3
## Alinhamento com os Requisitos Obrigatórios do TC_F3

---

## Requisitos Obrigatórios do TC — Status de Atendimento

### 1. Fine-tuning de LLM com dados médicos internos

| Requisito TC | Status | Como foi implementado |
|---|---|---|
| Fine-tuning de um LLM (LLaMA, Falcon ou outro) | ✅ | Phi-3-mini-4k-instruct (3.8B) com QLoRA 4-bit via Unsloth |
| Usar protocolos médicos, perguntas frequentes e laudos | ✅ | MedQuAD: 5.274 pares Q&A médicos de 12 fontes NIH/CDC/NCI |
| Preprocessing, anonimização e curadoria dos dados | ✅ | `fase1_dados/preprocess.py` + `medquad_dataset/clean_medquad.py` |
| Dataset de treinamento preparado | ✅ | `data/train.jsonl` (4.747) + `data/val.jsonl` (527 amostras) |

### 2. Criação de assistente médico com LangChain

| Requisito TC | Status | Como foi implementado |
|---|---|---|
| Usar LangChain para construir pipeline com LLM customizada | ✅ | `fase3_langchain/chains.py` — pipeline RAG com FAISS + LangChain |
| Realizar consultas em base de dados estruturadas | ✅ | Vector store FAISS com MedQuAD PT-BR indexado |
| Contextualizar respostas com informações atualizadas | ✅ | RAG recupera top-3 documentos relevantes do MedQuAD por consulta |
| Fluxos de decisão automatizados (LangGraph) | ✅ | `fase3_langchain/graph.py` — 6 nós: entrada → triagem → RAG → geração → validação |

### 3. Segurança e validação

| Requisito TC | Status | Como foi implementado |
|---|---|---|
| Limites de atuação — nunca prescrever sem validação humana | ✅ | `fase4_seguranca/safety_guard.py` — flag de prescrição + aviso obrigatório |
| Logging detalhado para rastreamento e auditoria | ✅ | `fase4_seguranca/logging_audit.py` → `logs/audit.log` em JSON |
| Explainability — indicar fonte da informação utilizada | ✅ | `fase4_seguranca/explainability.py` — fontes MedQuAD + score por resposta |

### 4. Organização do código

| Requisito TC | Status | Como foi implementado |
|---|---|---|
| Projeto modularizado em Python | ✅ | 5 fases separadas em pastas + módulos independentes |
| Instruções completas no README | ✅ | README.md com instalação, configuração e fluxo sequencial |

---

## Entregáveis da Fase 3 — Status

### Repositório Git

| Entregável | Status | Arquivo/Localização |
|---|---|---|
| Pipeline de fine-tuning | ✅ | `fase2_finetuning/MedAssist_FineTuning_Colab.ipynb` |
| Integração com LangChain | ✅ | `fase3_langchain/chains.py` + `assistant.py` |
| Fluxos do LangGraph | ✅ | `fase3_langchain/graph.py` |
| Dataset anonimizado | ✅ | `data/medquad_ptbr.jsonl` (MedQuAD — dataset público NIH) |
| Relatório técnico detalhado | ✅ | `outputs/relatorio_tecnico_final.md` |
| Explicação do processo de fine-tuning | ✅ | Notebook Colab + README.md seção Fine-tuning |
| Descrição do assistente médico | ✅ | README.md + documento de entrega |
| Diagrama do fluxo LangChain/LangGraph | ✅ | `outputs/langgraph_diagram.txt` + documento de entrega |
| Avaliação do modelo e análise de resultados | ✅ | `fase5_avaliacao/` + `outputs/relatorio_tecnico_final.md` |

### Dataset sugerido pelo TC

| Dataset | Status | Como foi usado |
|---|---|---|
| MedQuAD (sugerido pelo TC) | ✅ | Dataset principal — 5.274 pares Q&A em PT-BR |
| PubMedQA (sugerido pelo TC) | ⬜ | Não utilizado (MedQuAD suficiente para o escopo) |

---

## Checklist de Instalação e Execução

### FASE 0 — Setup

- [ ] `python -m venv venv && source venv/bin/activate` (Mac/Linux) ou `venv\Scripts\activate` (Windows)
- [ ] `pip install -r requirements.txt`
- [ ] `cp .env.example .env` e configurar MODO 1 ou MODO 2
- [ ] Confirmar `outputs/model/adapter_model.safetensors` (114 MB) presente

**Modos disponíveis:**
```
MODO 1 (OpenAI):   LLM_BACKEND=openai + OPENAI_API_KEY=sk-...
MODO 2 (Ollama):   OPENAI_BASE_URL=http://localhost:11434/v1 + OPENAI_MODEL=gemma3:4b
```

---

### FASE 1 — Dados

- [ ] `python fase1_dados/explore_dataset.py` → `outputs/explore_report.txt`
- [ ] `python fase1_dados/preprocess.py` → `data/train.jsonl` + `data/val.jsonl`
- [ ] `python fase1_dados/validate_data.py` → **APROVADA**

---

### FASE 2 — Fine-tuning (Google Colab)

- [ ] Abrir `fase2_finetuning/MedAssist_FineTuning_Colab.ipynb` no Colab
- [ ] Ativar GPU T4: Runtime → Change runtime type → T4 GPU
- [ ] Executar todas as células (~45 min)
- [ ] Baixar ZIP dos adaptadores → extrair em `outputs/model/`
- [ ] `python fase2_finetuning/validate_adapters.py` → **APROVADO 28/28**

---

### FASE 3 — Assistente

#### 3.1 Pipeline RAG
- [ ] `python fase3_langchain/chains.py` — vector store FAISS criado
- [ ] Verificar `outputs/vectorstore/index.faiss` gerado
- [ ] Respostas com fontes MedQuAD visíveis

#### 3.2 Fluxo LangGraph
- [ ] `python fase3_langchain/graph.py`
- [ ] Testar: consulta, emergência (SAMU 192), fora do escopo, prescrição

#### 3.3 Interface Web
- [ ] `python web_app.py` → http://localhost:8080
- [ ] Verificar: sidebar, chat, badges, fontes, painel de detalhes
- [ ] Testar emergência → bolha vermelha + SAMU 192
- [ ] Clicar "Configurar Backend" → alternar entre MODO 1 e MODO 2

#### 3.4 Ollama (MODO 2) — se aplicável
- [ ] Ollama instalado e `gemma3:4b` baixado
- [ ] `ollama serve` rodando (Linux)
- [ ] `.env` configurado com `OPENAI_BASE_URL=http://localhost:11434/v1`

---

### FASE 4 — Segurança

- [ ] `python fase4_seguranca/safety_guard.py` → 5/5 cenários corretos
- [ ] `python fase4_seguranca/logging_audit.py` → `logs/audit.log` JSON gerado
- [ ] `python fase4_seguranca/explainability.py` → score + fontes por resposta

---

### FASE 5 — Avaliação

- [ ] `python fase5_avaliacao/benchmark.py`
- [ ] `python fase5_avaliacao/generate_report.py`
- [ ] `outputs/relatorio_tecnico_final.md` completo

---

## Checklist de Submissão

**Código-fonte:**
- [ ] `web_app.py` + `web_ui.html` — interface web funcional
- [ ] `fase3_langchain/llm_backend.py` — 2 backends (OpenAI e Ollama)
- [ ] `fase3_langchain/chains.py`, `graph.py`, `assistant.py`
- [ ] `fase4_seguranca/` — safety, logging, explainability
- [ ] `fase5_avaliacao/` — benchmark, relatório
- [ ] `medquad_dataset/` — scripts de tratamento do dataset

**Dados e modelos:**
- [ ] `outputs/model/` — adaptadores LoRA (28/28 validações OK)
- [ ] `data/medquad_ptbr.jsonl` — 5.274 amostras PT-BR
- [ ] `outputs/relatorio_tecnico_final.md`
- [ ] `logs/audit.log` — log de interações gerado

**Documentação:**
- [ ] `README.md` — instruções completas dos 2 modos
- [ ] `CHECKLIST.md` — este arquivo
- [ ] `OLLAMA_SETUP.md` — guia do MODO 2
- [ ] `medquad_dataset/README.md` — documentação do pipeline de dados
- [ ] `.env.example` — template com os 2 modos

**Entregável pendente:**
- [ ] Vídeo demonstrativo (até 15 min)
  - Interface web + perguntas clínicas
  - Troca de backend pelo modal
  - Emergência → SAMU 192
  - Painel de detalhes e fontes MedQuAD
  - Fine-tuning no Colab + adaptadores

---

## Observações

- Nunca versione o `.env` real (contém chaves secretas)
- O `local_server.py` carrega o Phi-3-mini + LoRA — requer GPU (~8GB VRAM)
- O MODO 2 (Ollama gemma3:4b) funciona sem GPU, mas é mais lento em CPU
- O dataset MedQuAD é público (NIH) — não contém dados de pacientes reais
