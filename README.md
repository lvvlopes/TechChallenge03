# MedAssist — Assistente Médico Inteligente com IA
### Tech Challenge IADT — Fase 3 | POSTECH / FIAP

Assistente virtual médico com **fine-tuning real de LLM** (Phi-3-mini + QLoRA, Google Colab T4), **RAG sobre MedQuAD PT-BR** (5.274 amostras), orquestração via **LangChain + LangGraph**, guardrails de segurança clínica e interface web profissional com **três backends de inferência** configuráveis.

---

## Sumário

1. [Visão Geral](#visão-geral)
2. [Arquitetura](#arquitetura)
3. [Pré-requisitos](#pré-requisitos)
4. [Instalação passo a passo](#instalação-passo-a-passo)
5. [Configuração do Backend](#configuração-do-backend)
   - [MODO 1 — OpenAI API](#modo-1--openai-api)
   - [MODO 2 — Ollama + Gemma 3](#modo-2--ollama--gemma-3-local-gemma34b)
   - [MODO 3 — Phi-3-mini fine-tunado](#modo-3--phi-3-mini-fine-tunado-via-ollama-sem-gpu)
6. [Rodando o projeto](#rodando-o-projeto)
7. [Estrutura do projeto](#estrutura-do-projeto)
8. [Fine-tuning no Google Colab](#fine-tuning-no-google-colab)
9. [Pipeline de Dados — MedQuAD](#pipeline-de-dados--medquad)
10. [Segurança e Guardrails](#segurança-e-guardrails)
11. [Avaliação](#avaliação)
12. [Troubleshooting](#troubleshooting)

---

## Visão Geral

O MedAssist foi desenvolvido para o Tech Challenge Fase 3 da POSTECH/FIAP. O desafio é criar um **assistente virtual médico treinado com dados próprios**, capaz de auxiliar em condutas clínicas e responder dúvidas com base em protocolos médicos, organizando fluxos de decisão automatizados e seguros coordenados com LangChain.

**Tecnologias principais:**

| Componente | Tecnologia |
|---|---|
| Fine-tuning | Phi-3-mini-4k-instruct + QLoRA 4-bit (Unsloth) |
| Dados de treino | MedQuAD PT-BR — 5.274 pares médicos Q&A |
| RAG (Retrieval) | FAISS + sentence-transformers (MiniLM-L12) |
| Orquestração | LangChain + LangGraph |
| Backend de inferência | OpenAI API **ou** Ollama gemma3:4b **ou** Phi-3-mini fine-tunado (GGUF) |
| Segurança | Safety Guard, Logging/Audit, Explainability |
| Interface | FastAPI + HTML/CSS/JS (porta 8080) |

---

## Arquitetura

```
┌───────────────────────────────────────────┐
│    Interface Web — http://localhost:8080   │
│         (web_ui.html + FastAPI)            │
└──────────────────┬────────────────────────┘
                   │
       ┌───────────▼─────────────┐
       │   LangGraph Orchestrator │
       │                          │
       │  [entrada] → [triagem]   │
       │       ↓            ↓     │
       │  [emergência]  [busca_rag]← FAISS + MedQuAD PT-BR
       │       ↓            ↓     │
       │       └──→[geração]←──┘  │← llm_backend.py
       │            ↓             │
       │       [validação]        │← Safety + Disclaimer
       └───────────┬──────────────┘
                   │
       ┌───────────▼─────────────┐
       │      llm_backend.py      │
       │  ┌──────────┬──────────┬────────────┐ │
       │  │ OpenAI   │  Ollama  │  Ollama    │ │
       │  │   API    │ gemma3:4b│phi3-medass.│ │
       │  └──────────┴──────────┴────────────┘ │
       └───────────┬──────────────┘
                   │
       ┌───────────▼─────────────┐
       │   Logging / Audit        │
       │   logs/audit.log (JSON)  │
       └──────────────────────────┘
```

---

## Pré-requisitos

- **Python 3.10+** (recomendado 3.11)
- **pip** atualizado
- **Git**
- Para MODO 1: Chave de API da OpenAI
- Para MODO 2: ~4 GB de espaço livre + Ollama instalado
- Para MODO 3: Exportar GGUF no Colab (Célula 12b/12c) + Ollama instalado + `ollama pull gemma3:4b`
- Para MODO 3: arquivo `phi3-medassist.gguf` exportado do Colab (Células 12b/12c) + Ollama instalado

> **Nota sobre GPU:** O fine-tuning foi realizado no Google Colab com GPU T4. Para inferência, os três modos funcionam **sem GPU** — o MODO 3 usa o próprio modelo treinado convertido para GGUF e rodado pelo Ollama em CPU.

---

## Instalação passo a passo

### 1. Clonar o repositório

```bash
git clone https://github.com/<seu-usuario>/medassist.git
cd medassist
```

### 2. Criar e ativar ambiente virtual

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Tempo estimado: 3–7 minutos.

### 4. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Edite o .env escolhendo MODO 1, MODO 2 ou MODO 3 (ver próxima seção)
```

### 5. Baixar os arquivos grandes do Google Drive

Dois arquivos são grandes demais para o Git e estão disponíveis via **Google Drive**:

| Arquivo | Tamanho | Necessário para |
|---|---|---|
| `outputs/model/adapter_model.safetensors` | ~114 MB | Todos os modos (validação do fine-tuning) |
| `outputs/gguf/phi-3-mini-4k-instruct.Q4_K_M.gguf` | ~2.2 GB | Apenas MODO 3 — Phi-3-mini fine-tunado |

> 📁 **Link do Google Drive:** `[link será disponibilizado pelo autor]`

#### Arquivo 1 — Adaptadores LoRA (`adapter_model.safetensors`)

1. Baixe o arquivo do Google Drive
2. Coloque em `outputs/model/`:

```
outputs/model/
├── adapter_model.safetensors   ← baixado do Drive (114 MB)
├── adapter_config.json
├── tokenizer.json
├── tokenizer.model
└── chat_template.jinja
```

#### Arquivo 2 — GGUF do Phi-3-mini fine-tunado — apenas MODO 3

1. Baixe o arquivo `phi-3-mini-4k-instruct.Q4_K_M.gguf` do Google Drive
2. Coloque em `outputs/gguf/`:

```
outputs/gguf/
├── phi-3-mini-4k-instruct.Q4_K_M.gguf   ← baixado do Drive (~2.2 GB)
└── Modelfile
```

> ⚠️ O arquivo `.gguf` só é necessário para o **MODO 3**. Para MODOs 1 e 2 pode ignorar.

### 6. Verificar adaptadores LoRA

```bash
python fase2_finetuning/validate_adapters.py
# Resultado esperado: APROVADO 28/28
```

---

## Configuração do Backend

Edite o arquivo `.env` para escolher o backend de inferência:

---

### MODO 1 — OpenAI API

**Mais simples. Sem GPU. Sem instalação extra. Requer chave da OpenAI.**

```env
LLM_BACKEND=openai
OPENAI_API_KEY=sk-...           # sua chave em https://platform.openai.com/api-keys
OPENAI_MODEL=gpt-4o-mini        # ou gpt-4o, gpt-3.5-turbo
OPENAI_BASE_URL=                # deixe vazio
```

---

### MODO 2 — Ollama + Gemma 3 local (gemma3:4b)

**Sem API key. Sem custo. Funciona offline.**

#### Passo 2.1 — Instalar o Ollama

**Windows:**
```powershell
winget install Ollama.Ollama
# ou baixe em: https://ollama.com/download/windows
```

**Mac:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
# Verificar instalação
ollama --version
```

#### Passo 2.2 — Baixar o Gemma 3 (4B)

```bash
ollama pull gemma3:4b
# Download de ~3.3 GB — acontece apenas uma vez
```

#### Passo 2.3 — Iniciar o servidor Ollama

```bash
# Windows/Mac: inicia automaticamente após instalação
# Linux: iniciar manualmente:
ollama serve

# Verificar se está rodando:
curl http://localhost:11434/api/tags
```

#### Passo 2.4 — Configurar o .env

```env
LLM_BACKEND=openai
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=gemma3:4b
```

> Guia completo: [OLLAMA_SETUP.md](OLLAMA_SETUP.md)

---

### MODO 3 — Phi-3-mini fine-tunado via Ollama (sem GPU)

**Usa o seu próprio modelo treinado no Colab, convertido para GGUF e registrado no Ollama. Sem API key. Sem GPU. Funciona offline. Máxima especialização médica.**

Este é o modo que conecta o fine-tuning diretamente à inferência local.

#### Como funciona

```
Colab: Phi-3-mini + LoRA → fusão → GGUF quantizado (q4_k_m ~2.2 GB)
                                          ↓
                              ollama create phi3-medassist
                                          ↓
                         http://localhost:11434/v1  ←  web_app.py
```

#### Passo 3.1 — Exportar o GGUF no Colab

Abra o notebook `fase2_finetuning/MedAssist_FineTuning_Colab.ipynb` e execute as células **após o treinamento**:

- **Célula 12b** — funde LoRA + modelo base e exporta `phi-3-mini-4k-instruct.Q4_K_M.gguf` (~2.2 GB)
- **Célula 12c** — faz o download do arquivo para sua máquina

> O Colab precisa estar com GPU T4 ativa para fazer a fusão. Após isso, o GGUF roda localmente sem GPU.

#### Passo 3.2 — Mover o arquivo para o projeto

**Windows (PowerShell):**
```powershell
mkdir outputs\gguf
move $env:USERPROFILE\Downloads\phi-3-mini-4k-instruct.Q4_K_M.gguf outputs\gguf\phi3-medassist.gguf
```

**Mac/Linux:**
```bash
mkdir -p outputs/gguf
mv ~/Downloads/phi-3-mini-4k-instruct.Q4_K_M.gguf outputs/gguf/phi3-medassist.gguf
```

#### Passo 3.3 — Criar o Modelfile

```bash
# Windows (PowerShell):
@"
FROM ./phi3-medassist.gguf

SYSTEM "Você é um assistente médico especializado. Responda de forma precisa, clara e em português. Sempre recomende consulta com profissional de saúde para diagnósticos e tratamentos."

PARAMETER temperature 0.3
PARAMETER num_ctx 2048
"@ | Out-File -Encoding utf8 outputs\gguf\Modelfile

# Mac/Linux:
cat > outputs/gguf/Modelfile << 'EOF'
FROM ./phi3-medassist.gguf

SYSTEM "Você é um assistente médico especializado. Responda de forma precisa, clara e em português. Sempre recomende consulta com profissional de saúde para diagnósticos e tratamentos."

PARAMETER temperature 0.3
PARAMETER num_ctx 2048
EOF
```

#### Passo 3.4 — Registrar no Ollama

```bash
cd outputs/gguf
ollama create phi3-medassist -f Modelfile
# Aguarde 1-2 minutos

# Verificar
ollama list
# Deve aparecer: phi3-medassist

# Testar diretamente
ollama run phi3-medassist "Quais são os sintomas do diabetes tipo 2?"
```

#### Passo 3.5 — Configurar o .env

```env
LLM_BACKEND=openai
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=phi3-medassist
```

Ou selecione **MODO 3 — Phi-3-mini fine-tunado** no modal "Configurar Backend" da interface web.

---

### Comparativo dos três modos

| | MODO 1 | MODO 2 | MODO 3 |
|---|---|---|---|
| Modelo | gpt-4o-mini | gemma3:4b | **phi3-medassist** |
| Fine-tuning médico | Não | Não | **Sim (seu Colab)** |
| GPU necessária | Não | Não | Não |
| API key | Sim | Não | Não |
| Funciona offline | Não | Sim | Sim |
| Tamanho do modelo | — (nuvem) | ~3.3 GB | ~2.2 GB |
| Velocidade (CPU) | ~2s | ~15-30s | ~20-40s |

---

### MODO 3 — Phi-3-mini fine-tunado via Ollama (sem GPU)

**O modelo treinado no Colab rodando localmente. Sem GPU. Sem API key. Máxima especialização médica.**

Este modo usa o **mesmo modelo fine-tunado no Colab** (Phi-3-mini + QLoRA), convertido para GGUF e registrado no Ollama. É a única forma de rodar seus adaptadores LoRA sem GPU.

#### Passo 3.1 — Exportar o GGUF no Colab

No notebook `fase2_finetuning/MedAssist_FineTuning_Colab.ipynb`:
- Execute a **Célula 12b** → funde LoRA + modelo base e exporta GGUF (~2.2 GB)
- Execute a **Célula 12c** → baixa o arquivo `.gguf`

#### Passo 3.2 — Mover o arquivo para o projeto

```bash
mkdir -p outputs/gguf
mv ~/Downloads/phi-3-mini-4k-instruct.Q4_K_M.gguf outputs/gguf/phi3-medassist.gguf
```

#### Passo 3.3 — Criar o Modelfile

```bash
cat > outputs/gguf/Modelfile << 'EOF'
FROM ./phi3-medassist.gguf

SYSTEM "Você é um assistente médico especializado. Responda de forma precisa, clara e em português. Sempre recomende consulta com profissional de saúde."

PARAMETER temperature 0.3
PARAMETER num_ctx 2048
EOF
```

#### Passo 3.4 — Registrar no Ollama

```bash
cd outputs/gguf
ollama create phi3-medassist -f Modelfile

# Verificar:
ollama list   # phi3-medassist deve aparecer
```

#### Passo 3.5 — Configurar o .env

```env
LLM_BACKEND=openai
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=phi3-medassist
```

> **Comparativo dos três modos:**
>
> | Modo | Modelo | Especialização | GPU | Custo |
> |---|---|---|---|---|
> | MODO 1 | gpt-4o-mini | Geral + RAG | Não | Por token |
> | MODO 2 | gemma3:4b | Geral + RAG | Não | Gratuito |
> | **MODO 3** | **phi3-medassist** | **Fine-tuned + RAG** | **Não** | **Gratuito** |


---

### Trocar o backend pela interface web (sem reiniciar)

1. Abrir **http://localhost:8080**
2. Clicar em **"Configurar Backend"** no header
3. Escolher **MODO 1** (OpenAI), **MODO 2** (Ollama gemma3:4b) ou **MODO 3** (Phi-3-mini fine-tunado)
4. Preencher os campos e clicar em **"Aplicar configuração"**

---

## Rodando o projeto

### Fluxo sequencial completo (primeira execução)

```bash
# ── FASE 1: Dados ──────────────────────────────────────
python fase1_dados/explore_dataset.py     # → outputs/explore_report.txt
python fase1_dados/preprocess.py          # → data/train.jsonl + data/val.jsonl
python fase1_dados/validate_data.py       # → APROVADA

# ── FASE 2: Fine-tuning ────────────────────────────────
# (já realizado no Colab — adaptadores em outputs/model/)
python fase2_finetuning/validate_adapters.py  # → APROVADO 28/28

# ── FASE 3: Assistente ─────────────────────────────────
python fase3_langchain/chains.py          # → cria outputs/vectorstore/
python fase3_langchain/graph.py           # → testa o fluxo LangGraph
python web_app.py                         # → http://localhost:8080

# ── FASE 4: Segurança ──────────────────────────────────
python fase4_seguranca/safety_guard.py    # → 5/5 cenários corretos
python fase4_seguranca/logging_audit.py   # → logs/audit.log gerado
python fase4_seguranca/explainability.py  # → score + fontes

# ── FASE 5: Avaliação ──────────────────────────────────
python fase5_avaliacao/benchmark.py
python fase5_avaliacao/generate_report.py # → outputs/relatorio_tecnico_final.md
```

### Iniciar somente a interface (execuções seguintes)

```bash
python web_app.py
# Acesse: http://localhost:8080
```

---

## Estrutura do projeto

```
medassist/
│
├── web_app.py                   # Servidor FastAPI (porta 8080)
├── web_ui.html                  # Interface web profissional
├── config.py                    # Configurações globais
├── utils.py                     # Utilitários gerais
├── local_server.py              # Servidor Phi-3-mini OpenAI-compatible (GPU)
├── requirements.txt             # Dependências Python
├── .env.example                 # Template — MODO 1 (OpenAI), MODO 2 (Ollama) e MODO 3 (phi3-medassist)
│
├── data/
│   ├── medquad_ptbr.jsonl       # 5.274 pares médicos Q&A em PT-BR
│   ├── train.jsonl              # 4.747 amostras de treino (90%)
│   └── val.jsonl                # 527 amostras de validação (10%)
│
├── medquad_dataset/
│   ├── README.md                # Documentação completa do pipeline de dados
│   ├── MedQuAD/                 # Dataset original clonado do GitHub (NIH)
│   ├── clean_medquad.py         # Script de limpeza XML → JSONL
│   ├── translate_ptbr.py        # Script de tradução EN → PT-BR
│   ├── medquad_clean.jsonl      # Dataset limpo em inglês (8.653 pares)
│   └── medquad_ptbr.jsonl       # Dataset traduzido PT-BR (5.274 pares)
│
├── outputs/
│   ├── gguf/                    # MODO 3: modelo GGUF + Modelfile (gerado no Colab)
│   │   ├── phi-3-mini-4k-instruct.Q4_K_M.gguf  ← baixar do Google Drive (~2.2 GB)
│   │   └── Modelfile            ← config do Ollama para o modelo fine-tunado
│   ├── model/                   # Adaptadores LoRA do fine-tuning
│   │   ├── adapter_model.safetensors      ← baixar do Google Drive (114 MB)
│   │   ├── adapter_config.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer.model
│   │   └── chat_template.jinja
│   ├── vectorstore/             # Vector store FAISS para RAG
│   │   ├── index.faiss
│   │   └── index.pkl
│   └── relatorio_tecnico_final.md
│
├── fase1_dados/
│   ├── explore_dataset.py       # Análise exploratória
│   ├── preprocess.py            # Preprocessing e split train/val
│   └── validate_data.py         # Validação de qualidade
│
├── fase2_finetuning/
│   ├── MedAssist_FineTuning_Colab.ipynb  # Notebook Google Colab
│   └── validate_adapters.py     # Valida adaptadores LoRA (28/28)
│
├── fase3_langchain/
│   ├── llm_backend.py           # Backend unificado (OpenAI / Ollama)
│   ├── chains.py                # Pipeline RAG com FAISS
│   ├── graph.py                 # Fluxo LangGraph
│   └── assistant.py             # Interface do assistente
│
├── fase4_seguranca/
│   ├── safety_guard.py          # Guardrails clínicos
│   ├── logging_audit.py         # Logging JSON
│   └── explainability.py        # Fontes e score de confiança
│
├── fase5_avaliacao/
│   ├── benchmark.py             # Métricas ROUGE, BLEU, F1
│   └── generate_report.py       # Relatório técnico final
│
├── logs/
│   └── audit.log                # Log de interações (JSON)
│
├── README.md                    # Este arquivo
├── CHECKLIST.md                 # Checklist alinhado ao TC Fase 3
└── OLLAMA_SETUP.md              # Guia completo do MODO 2 e MODO 3 (Ollama)
```

---

## Fine-tuning no Google Colab

O fine-tuning foi realizado com **QLoRA 4-bit** via Unsloth no Google Colab (GPU T4 gratuita).

### Como executar o fine-tuning

1. Abrir: `fase2_finetuning/MedAssist_FineTuning_Colab.ipynb` no Google Colab
2. Ativar GPU: **Runtime → Change runtime type → T4 GPU**
3. Executar todas as células (~45 minutos)
4. Baixar o ZIP dos adaptadores gerados pelo Colab
5. Extrair em `outputs/model/`

### Parâmetros do treinamento

| Parâmetro | Valor |
|---|---|
| Modelo base | Phi-3-mini-4k-instruct (3.8B parâmetros) |
| Técnica | QLoRA 4-bit (NF4) via Unsloth |
| LoRA rank / alpha | r=16 / α=32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Dataset | 4.747 amostras MedQuAD PT-BR |
| Épocas | 1 (~45 min T4) |
| Batch efetivo | 16 (gradient accumulation 4 × batch 4) |
| Learning rate | 2e-4 |
| Otimizador | AdamW 8-bit |

### Validação dos adaptadores

```bash
python fase2_finetuning/validate_adapters.py
# APROVADO 28/28 verificações
```

---

## Pipeline de Dados — MedQuAD

O dataset **MedQuAD** (NIH) foi processado em três etapas para gerar os dados de fine-tuning:

| Etapa | Script | Entrada | Saída |
|---|---|---|---|
| 1. Clone | `git clone` | GitHub | ~16.000 XMLs brutos |
| 2. Limpeza | `clean_medquad.py` | XMLs | 8.653 pares EN (JSONL) |
| 3. Tradução PT-BR | `translate_ptbr.py` | 8.653 pares EN | 5.274 pares PT-BR (JSONL) |

Documentação completa: [medquad_dataset/README.md](medquad_dataset/README.md)

---

## Segurança e Guardrails

| Guardrail | Trigger | Ação |
|---|---|---|
| Emergência médica | Infarto, AVC, parada cardíaca... | Bloqueia LLM → SAMU 192 / Bombeiros 193 |
| Saúde mental crítica | Suicídio, automutilação | Redireciona → CVV 188 |
| Prescrição direta | Solicitação de medicação | Aviso: revisão humana obrigatória |
| Fora do escopo | Saudações, tópicos não médicos | Resposta padrão de escopo |
| Disclaimer | Toda resposta | Aviso de não substituição profissional |
| Logging | Toda interação | `logs/audit.log` em formato JSON |
| Explainability | Toda resposta | Fontes MedQuAD + score de confiança |

---

## Avaliação

```bash
python fase5_avaliacao/benchmark.py
python fase5_avaliacao/generate_report.py
# → outputs/relatorio_tecnico_final.md
```

---

## Troubleshooting

**`ModuleNotFoundError`**
```bash
pip install -r requirements.txt
```

**`OPENAI_API_KEY não configurada`**
```bash
cp .env.example .env  # depois edite o .env
```

**Ollama não responde**
```bash
ollama serve          # Linux: inicia o servidor
ollama list           # verifica modelos baixados
                      # MODO 2: deve aparecer gemma3:4b
                      # MODO 3: deve aparecer phi3-medassist
```

**MODO 3 — phi3-medassist não aparece no `ollama list`**
```bash
# Registre o modelo a partir do GGUF exportado do Colab:
cd outputs/gguf
ollama create phi3-medassist -f Modelfile

# Se ainda não tiver o .gguf, exporte no Colab:
# Abra MedAssist_FineTuning_Colab.ipynb → execute Células 12b e 12c
```

**Vector store não encontrado**
```bash
python fase3_langchain/chains.py
```

**Adaptadores LoRA não encontrados**
```
outputs/model/adapter_model.safetensors não existe?
→ Baixe do Google Drive e coloque em outputs/model/
   Link: [link será disponibilizado pelo autor]
```

---

> ⚠️ **Aviso médico:** Este sistema é informativo e educacional. Não substitui avaliação de profissional de saúde habilitado. Em emergências, ligue **SAMU 192** ou **Bombeiros 193**.
