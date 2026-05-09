# MedQuAD Dataset — Pipeline de Tratamento e Tradução PT-BR

## Visão Geral

O **MedQuAD (Medical Question Answering Dataset)** é um corpus público de perguntas e respostas médicas em inglês, organizado pelo NIH (National Institutes of Health) dos Estados Unidos. Este diretório documenta o pipeline completo realizado para transformar o dataset bruto em amostras de fine-tuning em português brasileiro.

**Resultado final:** 5.274 pares pergunta-resposta médicos em PT-BR, prontos para fine-tuning de LLM.

---

## Estrutura do Dataset Original

O MedQuAD contém pares de perguntas e respostas extraídos de 12 categorias de fontes médicas oficiais:

| Pasta | Fonte | Descrição |
|---|---|---|
| `01_MedlinePlus_Health_Topic_QA/` | MedlinePlus | Tópicos gerais de saúde |
| `02_GARD_QA/` | GARD (NIH) | Doenças raras |
| `03_GHR_QA/` | Genetics Home Reference | Condições genéticas |
| `04_MPlus_ADAM_QA/` | MedlinePlus ADAM | Enciclopédia médica |
| `05_NIDDK_QA/` | NIDDK (NIH) | Diabetes, digestão, rim |
| `06_NINDS_QA/` | NINDS (NIH) | Doenças neurológicas |
| `07_SeniorHealth_QA/` | SeniorHealth | Saúde do idoso |
| `08_NHLBI_QA/` | NHLBI (NIH) | Coração e pulmão |
| `09_CDC_QA/` | CDC | Controle de doenças |
| `10_MPlus_ADAM_QA/` | MedlinePlus ADAM | Enciclopédia médica |
| `11_MPlusDrugs_QA/` | MedlinePlus Drugs | Medicamentos |
| `12_Cancer_QA/` | NCI | Oncologia |

Total de arquivos XML no dataset original: ~16.000 pares brutos.

---

## Pipeline de Tratamento

### Etapa 1 — Clone do Dataset

```bash
git clone https://github.com/abachaa/MedQuAD.git
```

O dataset é clonado do repositório oficial no GitHub. Cada arquivo XML contém pares `<Question>` / `<Answer>` estruturados.

### Etapa 2 — Limpeza e Filtragem (`clean_medquad.py`)

O script percorre todos os arquivos XML e aplica as seguintes regras de qualidade:

**Parsing XML:**
```python
for root, dirs, files in os.walk("MedQuAD"):
    for file in files:
        if file.endswith(".xml"):
            tree = ET.parse(path)
            # Extrai pares QAPair -> Question / Answer
```

**Filtros de qualidade aplicados:**
| Filtro | Critério | Motivo |
|---|---|---|
| Tamanho mínimo da pergunta | `len(q) >= 10` | Remove perguntas degeneradas |
| Tamanho mínimo da resposta | `len(a) >= 20` | Remove respostas vazias |
| Tamanho máximo da resposta | `len(a) <= 1000` | Remove respostas excessivamente longas |
| Conteúdo inválido | `"contact your doctor" not in a` | Remove respostas genéricas sem valor |
| Deduplicação | Hash por conteúdo da pergunta | Remove duplicatas exactas |
| Limpeza de HTML | `re.sub(r'<[^>]+>', '', text)` | Remove tags HTML residuais |
| Normalização de espaços | `re.sub(r'\s+', ' ', text)` | Normaliza espaços em branco |

**Formato de saída (`medquad_clean.jsonl`):**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful medical assistant."},
    {"role": "user",   "content": "What are the symptoms of type 2 diabetes?"},
    {"role": "assistant", "content": "Type 2 diabetes symptoms include..."}
  ]
}
```

**Resultado da limpeza:** 16.000+ pares brutos → **8.653 pares válidos** em inglês.

### Etapa 3 — Tradução para PT-BR (`translate_ptbr.py`)

A tradução foi realizada usando a biblioteca `deep_translator` com o motor Google Translate:

```python
from deep_translator import GoogleTranslator
translator = GoogleTranslator(source='en', target='pt')
```

**O que foi traduzido:**
- Pergunta do usuário (campo `user`)
- Resposta do assistente (campo `assistant`)
- O prompt de sistema foi reescrito diretamente em PT-BR: `"Você é um assistente médico útil."`

**Formato de saída (`medquad_ptbr.jsonl`):**
```json
{
  "messages": [
    {"role": "system",    "content": "Você é um assistente médico útil."},
    {"role": "user",      "content": "Quais são os sintomas do diabetes tipo 2?"},
    {"role": "assistant", "content": "Os sintomas do diabetes tipo 2 incluem..."}
  ]
}
```

**Resultado da tradução:** 8.653 pares em inglês → **5.274 pares em PT-BR** (após segundo filtro de qualidade pós-tradução aplicado no preprocessing).

> **Nota:** A diferença de 8.653 → 5.274 deve-se a falhas de tradução (timeout, textos muito longos) e ao segundo filtro de qualidade aplicado durante o preprocessing da Fase 1 (`fase1_dados/preprocess.py`).

---

## Arquivos Gerados

| Arquivo | Tamanho | Descrição |
|---|---|---|
| `medquad_clean.jsonl` | ~8.653 linhas | Dataset limpo em inglês |
| `medquad_ptbr.jsonl` | ~5.274 linhas | Dataset traduzido para PT-BR |

Os arquivos também estão disponíveis em:
- `data/medquad_ptbr.jsonl` — cópia para uso no pipeline principal
- `data/train.jsonl` — 4.747 amostras (90% para fine-tuning)
- `data/val.jsonl` — 527 amostras (10% para validação)

---

## Como Reproduzir

```bash
# 1. Clonar o dataset original
git clone https://github.com/abachaa/MedQuAD.git medquad_dataset/MedQuAD

# 2. Instalar dependência de tradução
pip install deep-translator

# 3. Gerar dataset limpo em inglês
python medquad_dataset/clean_medquad.py
# Saída: medquad_dataset/medquad_clean.jsonl

# 4. Traduzir para PT-BR
python medquad_dataset/translate_ptbr.py
# Saída: medquad_dataset/medquad_ptbr.jsonl
# Tempo estimado: ~2-4 horas (dependendo da conexão e rate limits do Google)
```

> **Atenção:** A tradução via Google Translate possui rate limiting. Em caso de falha, o script mantém o texto original (fallback). Para datasets maiores, considere usar a API oficial do Google Translate com chave.

---

## Qualidade dos Dados

Relatório gerado pelo script de exploração (`fase1_dados/explore_dataset.py`):

```
Total de amostras:        5.274
Comprimento médio pergunta: 67 caracteres
Comprimento médio resposta: 312 caracteres
Amostras sem resposta:    0
Duplicatas:               0
```

O relatório completo está em `outputs/explore_report.txt` e `outputs/data_quality_report.txt`.

---

## Referência

Ben Abacha, A., & Zweigenbaum, P. (2015). MEANS: A medical question-answering system combining NLP techniques and semantic Web technologies. *Information Processing & Management*, 51(5), 570-594.

Dataset original: https://github.com/abachaa/MedQuAD
