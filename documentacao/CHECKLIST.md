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
| Instruções completas no README | ✅ | README.md com instalação, 3 modos, Google Drive e fluxo sequencial |

---

## Entregáveis da Fase 3 — Status

### Repositório Git

| Entregável | Status | Arquivo/Localização |
|---|---|---|
| Pipeline de fine-tuning | ✅ | `fase2_finetuning/MedAssist_FineTuning_Colab.ipynb` (inclui Células 12b/12c para exportar GGUF) |
| Integração com LangChain | ✅ | `fase3_langchain/chains.py` + `assistant.py` |
| Fluxos do LangGraph | ✅ | `fase3_langchain/graph.py` |
| Dataset anonimizado | ✅ | `data/medquad_ptbr.jsonl` (MedQuAD — dataset público NIH) |
| Relatório técnico detalhado | ✅ | `relatorio_tecnico_medassist.docx` + `outputs/relatorio_tecnico_final.md` |
| Explicação do processo de fine-tuning | ✅ | Notebook Colab + Seção 2 do relatório técnico |
| Descrição do assistente médico | ✅ | README.md + `relatorio_tecnico_medassist.docx` |
| Diagrama do fluxo LangChain/LangGraph | ✅ | `outputs/langgraph_diagram.txt` + `medassist_entrega.pdf` |
| Avaliação do modelo e análise de resultados | ✅ | `fase5_avaliacao/` + `outputs/relatorio_tecnico_final.md` |

### Arquivos grandes — Google Drive

> ⚠️ Os arquivos abaixo excedem o limite do Git e estão disponíveis via Google Drive.
> 📁 **Link:** `[link será disponibilizado pelo autor]` — ver README.md Passo 5

| Arquivo | Tamanho | Necessário para |
|---|---|---|
| `outputs/model/adapter_model.safetensors` | ~114 MB | Todos os modos (validação do fine-tuning — 28/28) |
| `outputs/gguf/phi-3-mini-4k-instruct.Q4_K_M.gguf` | ~2.2 GB | Apenas MODO 3 — Phi-3-mini fine-tunado |

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
- [ ] `cp .env.example .env` e configurar MODO 1, MODO 2 ou MODO 3
- [ ] Baixar `outputs/model/adapter_model.safetensors` (~114 MB) do Google Drive → ver README Passo 5

**Modos disponíveis:**
```
MODO 1 (OpenAI):          LLM_BACKEND=openai + OPENAI_API_KEY=sk-...
MODO 2 (Ollama gemma3):   OPENAI_BASE_URL=http://localhost:11434/v1 + OPENAI_MODEL=gemma3:4b
MODO 3 (phi3-medassist):  OPENAI_BASE_URL=http://localhost:11434/v1 + OPENAI_MODEL=phi3-medassist
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
- [ ] Executar todas as células de treinamento (~45 min)
- [ ] Adaptadores LoRA disponíveis via Google Drive → colocar em `outputs/model/`
- [ ] `python fase2_finetuning/validate_adapters.py` → **APROVADO 28/28**
- [ ] *(MODO 3)* Executar Células 12b e 12c → exportar e baixar o GGUF (~2.2 GB)

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
- [ ] Clicar "Configurar Backend" → verificar 3 opções disponíveis (MODO 1, 2 e 3)

#### 3.4 MODO 2 — Ollama + Gemma 3 (se utilizado)
- [ ] Ollama instalado e `gemma3:4b` baixado (`ollama pull gemma3:4b`)
- [ ] `ollama serve` rodando (Linux — automático no Win/Mac)
- [ ] `.env` configurado com `OPENAI_MODEL=gemma3:4b`

#### 3.5 MODO 3 — Phi-3-mini fine-tunado (se utilizado)
- [ ] `phi-3-mini-4k-instruct.Q4_K_M.gguf` baixado do Google Drive → `outputs/gguf/`
- [ ] Modelfile criado em `outputs/gguf/Modelfile`
- [ ] `cd outputs/gguf && ollama create phi3-medassist -f Modelfile`
- [ ] `ollama list` → confirmar `phi3-medassist` aparece
- [ ] `.env` configurado com `OPENAI_MODEL=phi3-medassist`

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
- [ ] `web_app.py` + `web_ui.html` — interface web com 3 backends configuráveis
- [ ] `fase3_langchain/llm_backend.py` — 3 modos (OpenAI, Ollama gemma3:4b, phi3-medassist)
- [ ] `fase3_langchain/chains.py`, `graph.py`, `assistant.py`
- [ ] `fase4_seguranca/` — safety, logging, explainability
- [ ] `fase5_avaliacao/` — benchmark, relatório
- [ ] `medquad_dataset/` — scripts de tratamento do dataset

**Dados e modelos:**
- [ ] `outputs/model/adapter_model.safetensors` — via Google Drive (~114 MB)
- [ ] `outputs/gguf/phi-3-mini-4k-instruct.Q4_K_M.gguf` — via Google Drive (~2.2 GB) *(MODO 3)*
- [ ] `outputs/gguf/Modelfile` — config do Ollama para o phi3-medassist *(MODO 3)*
- [ ] `data/medquad_ptbr.jsonl` — 5.274 amostras PT-BR
- [ ] `outputs/relatorio_tecnico_final.md`
- [ ] `logs/audit.log` — log de interações gerado

**Documentação:**
- [ ] `README.md` — instruções completas dos 3 modos + Google Drive
- [ ] `CHECKLIST.md` — este arquivo
- [ ] `OLLAMA_SETUP.md` — guia do MODO 2 (gemma3:4b) e MODO 3 (phi3-medassist)
- [ ] `medquad_dataset/README.md` — documentação do pipeline de dados
- [ ] `.env.example` — template com os 3 modos
- [ ] `relatorio_tecnico_medassist.docx` — relatório técnico completo
- [ ] `medassist_entrega.pdf` — documento de entrega com diagrama de arquitetura

**Entregável pendente:**
- [ ] Vídeo demonstrativo (até 15 min)
  - Interface web + perguntas clínicas
  - Troca de backend pelo modal (MODO 1, 2 e 3)
  - Emergência → SAMU 192
  - Painel de detalhes e fontes MedQuAD
  - Fine-tuning no Colab + exportação GGUF (Células 12b/12c)

---

## Observações

- Nunca versione o `.env` real (contém chaves secretas)
- `outputs/model/adapter_model.safetensors` e `outputs/gguf/*.gguf` estão no `.gitignore` — disponíveis via Google Drive
- O MODO 2 (Ollama gemma3:4b) funciona sem GPU, mas é mais lento em CPU
- O MODO 3 (phi3-medassist) usa o modelo treinado no Colab — mesma velocidade do MODO 2, maior especialização médica
- O dataset MedQuAD é público (NIH) — não contém dados de pacientes reais
- Para reproduzir o GGUF do zero: abrir o notebook no Colab e executar apenas as Células 12b e 12c (~5-10 min com GPU T4)
