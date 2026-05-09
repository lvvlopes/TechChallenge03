# Relatório Técnico Final — MedAssist
### Tech Challenge IADT — Fase 3 | POSTECH / FIAP
**Atualizado em:** 01/05/2026

---

## 1. Visão Geral do Projeto

O **MedAssist** é um assistente virtual médico desenvolvido como entregável do Tech Challenge Fase 3. O sistema integra fine-tuning real de LLM com dados médicos (MedQuAD PT-BR) e orquestração de fluxos de decisão via LangChain e LangGraph.

**Objetivo:** Criar um assistente capaz de auxiliar médicos e profissionais de saúde em condutas clínicas, respondendo dúvidas baseadas em protocolos médicos e sugerindo procedimentos com base em evidências — sempre com validação humana obrigatória.

**Arquitetura geral:**

```
[Usuário / Médico]
       │
       ▼
[LangGraph — Orquestrador de Fluxo]
       │
   ┌───┴─────────────────────────┐
   │                             │
[Triagem]                 [Safety Guard]
   │                             │
   ├── emergência ──► [Alerta]   │
   │                             │
   └── consulta ──► [RAG]        │
                     │           │
              [FAISS + MedQuAD]  │
                     │           │
              [OpenAI gpt-4o-mini]
                     │
        [Resposta + Fonte + Disclaimer]
                     │
              [Logging / Audit]
```

| Componente | Tecnologia |
|---|---|
| Fine-tuning | Phi-3-mini-4k-instruct + QLoRA via Unsloth (Google Colab T4) |
| Adaptadores LoRA | `outputs/model/` — 114 MB, gerados no Colab |
| LLM de inferência | OpenAI API (`gpt-4o-mini`) |
| RAG | FAISS + sentence-transformers |
| Dataset | MedQuAD PT-BR (5.274 amostras) |
| Orquestração | LangChain + LangGraph |
| Segurança | Safety Guard + Logging + Explainability |

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
============================================================
  📋 RELATÓRIO DE QUALIDADE DOS DADOS
============================================================

🚂 Treino:
   Registros:  4,747
   Perguntas — min:3 max:20 avg:8.2 palavras
   Respostas  — min:5 max:186 avg:84.2 palavras

🔬 Validação:
   Registros:  527
   Perguntas — min:3 max:22 avg:8.2 palavras
   Respostas  — min:7 max:179 avg:86.0 palavras

🔄 Data Leakage: 11 amostras sobrepostas

⚠️  Problemas encontrados (1):
   • AVISO: 11 perguntas do val existem no treino (data leakage).

============================================================
  ✅ Fase 1.3 — Validação APROVADA!
============================================================
```

---

## 3. Processo de Fine-tuning

### 3.1 Modelo Base

O fine-tuning foi realizado **exclusivamente no Google Colab com GPU T4** usando o modelo **Phi-3-mini-4k-instruct** (3.8B parâmetros) da Microsoft, com a biblioteca **Unsloth** para otimização.

- **Modelo:** `unsloth/Phi-3-mini-4k-instruct-bnb-4bit`
- **Motivo da escolha:** Modelo eficiente para a VRAM disponível na T4 (15 GB), excelente relação custo-benefício para tarefas de QA médico em PT-BR
- **Biblioteca:** Unsloth — 2× mais rápido que HuggingFace puro, 60% menos VRAM

### 3.2 Técnica: QLoRA via Unsloth

O fine-tuning utiliza **QLoRA (Quantized Low-Rank Adaptation)** — quantização 4-bit NF4 do modelo base combinada com adaptadores LoRA treináveis. Apenas os adaptadores são treinados (~0,5% dos parâmetros totais), mantendo os pesos originais do Phi-3-mini congelados.

```
Parâmetros LoRA:
  rank (r):       16
  alpha:          32   (escala = 2× rank)
  dropout:        0.05
  quantização:    4-bit NF4 (QLoRA)
  target modules: q_proj, k_proj, v_proj, o_proj,
                  gate_proj, up_proj, down_proj
```

O resultado são 224 pares lora_A/lora_B, totalizando 448 tensores e ~114 MB de artefato final.

### 3.3 Parâmetros de Treinamento

| Parâmetro | Valor |
|-----------|-------|
| Modelo base | Phi-3-mini-4k-instruct (3.8B) |
| Epochs | 1 (perfil rápido T4 free, ~45 min) |
| Batch size por device | 4 |
| Gradient accumulation | 4 steps |
| Batch efetivo | 16 |
| Learning rate | 2e-4 |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| Max seq length | 1.024 tokens |
| Ambiente | Google Colab T4 GPU (15 GB VRAM) |

### 3.4 Validação dos Adaptadores Gerados

A validação automática (`fase2_finetuning/validate_adapters.py`) verificou 28 critérios e obteve **28/28 aprovados**:

```
Resultado: APROVADO — 28/28 verificações

Arquivos:
  OK  adapter_config.json         (0.0 MB)
  OK  adapter_model.safetensors   (114.1 MB)
  OK  tokenizer.json              (3.5 MB)
  OK  tokenizer_config.json       (0.0 MB)
  OK  tokenizer.model (opcional)  (0.5 MB)
  OK  chat_template.jinja         (0.0 MB)

Configuração:
  OK  peft_type:     LORA
  OK  task_type:     CAUSAL_LM
  OK  lora rank (r): 16
  OK  lora_alpha:    32
  OK  target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  OK  inference_mode = True (pronto para uso)
  OK  base_model_name_or_path: microsoft/Phi-3-mini-4k-instruct

Pesos (safetensors):
  OK  Tamanho: 114.1 MB
  OK  448 tensores LoRA — 224 pares lora_A/lora_B balanceados
  OK  Prefixo correto: base_model.model
  OK  Todos os target_modules presentes
  OK  Formato: PyTorch

Tokenizer:
  OK  tokenizer.json — vocabulário: 32.000 tokens
  OK  eos_token, bos_token e pad_token definidos
  OK  tokenizer.model (sentencepiece): 488 KB
  OK  chat_template.jinja — 407 chars, tokens user/assistant presentes

Consistência:
  OK  target_modules batem entre config e pesos
  OK  inference_mode=True — adaptadores prontos para inferência
```

### 3.5 Avaliação Qualitativa das Respostas (Colab)

Após o treinamento, 5 perguntas médicas foram testadas diretamente no Colab com o modelo fine-tunado. Todas obtiveram respostas coerentes, estruturadas e em PT-BR:

| Pergunta | Resultado |
|----------|-----------|
| Quais são os sintomas da diabetes tipo 2? | 8 sintomas listados com descrição + orientação para consulta |
| O que causa hipertensão arterial? | 8 fatores causais (genéticos, comportamentais, ambientais) |
| Como funciona a quimioterapia? | Mecanismo, vias de administração, ciclos e efeitos colaterais |
| Quais são os sintomas do AVC? | 6 sinais com instrução de urgência ("buscar atendimento imediatamente") |
| O que é leucemia linfoblástica aguda? | Definição clínica, sintomas, epidemiologia e abordagem terapêutica |

**5/5 respostas aprovadas** — conteúdo clinicamente relevante, linguagem clara e adequada ao contexto médico brasileiro.

### 3.6 Decisão Arquitetural: Fine-tuning vs. Inferência em Produção

O Phi-3-mini fine-tunado requer GPU com ao menos 8 GB de VRAM para inferência em tempo real — recurso indisponível no ambiente local de desenvolvimento. A decisão arquitetural foi:

- **Fine-tuning (Colab):** Phi-3-mini + QLoRA — artefato técnico comprovando o processo de treinamento com dados médicos em português
- **Inferência em produção:** OpenAI API (`gpt-4o-mini`) + RAG sobre MedQuAD — garante disponibilidade, latência e qualidade de resposta independente de GPU local

Essa abordagem híbrida é padrão na indústria: treina-se um modelo especializado para demonstrar domínio técnico e personalização com dados do domínio, enquanto a inferência em produção usa uma API robusta com RAG para contextualização das respostas.

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

## 6. Avaliação

### 6.1 Validação dos Adaptadores LoRA

28/28 verificações aprovadas — detalhes completos na seção 3.4.

### 6.2 Avaliação Qualitativa do Modelo Fine-tunado

5/5 respostas aprovadas no teste realizado no Colab após o treinamento — detalhes na seção 3.5.

### 6.3 Avaliação do Sistema em Produção (RAG + OpenAI)

Testado via `logs/audit.log` com múltiplas interações reais, incluindo consultas clínicas, cenários de emergência e perguntas com menção a medicamentos.

**Pontos fortes:**
- Respostas fundamentadas no MedQuAD com citação de fontes em todas as interações
- Disclaimer médico presente em 100% das interações recentes
- Detecção de emergências funcional com redirecionamento correto para SAMU 192 e CVV 188
- Logging estruturado completo para rastreamento e auditoria

**Limitações e próximos passos:**
- Inferência com o Phi-3-mini fine-tunado requer ambiente com GPU — viável em produção com instância cloud
- Data leakage de 11 amostras pode ser corrigido com re-split estratificado
- Expansão do dataset com protocolos hospitalares internos aumentaria a especialização
- Avaliação humana por profissionais de saúde validaria a qualidade clínica das respostas

---

## 7. Conclusão

O MedAssist demonstra a viabilidade de construir um assistente médico em português com:

- **Fine-tuning real** de LLM (Phi-3-mini + QLoRA via Unsloth) com dados médicos em PT-BR, executado no Google Colab com GPU T4 e validado com 28/28 verificações aprovadas
- **RAG** sobre MedQuAD PT-BR para fundamentação das respostas em evidências, com citação de fontes em cada interação
- **Fluxos de decisão automatizados** com LangGraph cobrindo os cenários clínicos críticos
- **Guardrails robustos** para segurança clínica com logging completo para auditoria

O sistema adota uma arquitetura híbrida deliberada: os adaptadores LoRA do Phi-3-mini comprovam o processo de fine-tuning com dados do domínio médico, enquanto a inferência em produção via OpenAI + RAG garante disponibilidade e qualidade independente de infraestrutura de GPU. Essa separação permite evolução incremental — bastando substituir o endpoint de inferência por um servidor local com o modelo fine-tunado quando GPU estiver disponível.

---

*Relatório atualizado em 01/05/2026 com informações precisas do treinamento realizado no Google Colab (Phi-3-mini + QLoRA + Unsloth).*
