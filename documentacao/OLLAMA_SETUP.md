# Guia de Setup — Ollama
## MedAssist · Backends locais sem GPU obrigatória

O Ollama permite rodar modelos de linguagem localmente com uma API idêntica à OpenAI.
Sem API key, sem custo, sem dependência de nuvem.

O MedAssist suporta **dois modos locais via Ollama**:

| | MODO 2 | MODO 3 |
|---|---|---|
| Modelo | `gemma3:4b` | `phi3-medassist` |
| Fine-tuning médico | ❌ Modelo genérico | ✅ Seu modelo treinado no Colab |
| Download | `ollama pull gemma3:4b` | Google Drive + `ollama create` |
| Tamanho | ~3.3 GB | ~2.2 GB |
| GPU obrigatória | ❌ | ❌ |
| API key | ❌ | ❌ |
| Funciona offline | ✅ | ✅ |
| Custo | Gratuito | Gratuito |

---

## Passo 1 — Instalar o Ollama

**Windows:**
```powershell
# Baixe o instalador em: https://ollama.com/download/windows
# Ou via winget:
winget install Ollama.Ollama
```

**Mac:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verificar instalação:
```bash
ollama --version
```

---

## MODO 2 — Ollama + Gemma 3 (gemma3:4b)

Modelo genérico de uso geral. Especialização médica feita pelo RAG (MedQuAD PT-BR).

### Passo 2 — Baixar o Gemma 3 (4B)

```bash
# ~3.3 GB — acontece uma vez
ollama pull gemma3:4b
```

### Passo 3 — Iniciar o servidor Ollama

```bash
# Windows/Mac: inicia automaticamente após instalação.
# Linux: inicie manualmente:
ollama serve
```

Verificar se está rodando:
```bash
curl http://localhost:11434/api/tags
# Deve retornar JSON com gemma3:4b na lista
```

### Passo 4 — Configurar o .env

```env
LLM_BACKEND=openai
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=gemma3:4b
```

### Testar o MODO 2

```bash
# Via CLI do Ollama
ollama run gemma3:4b "Quais são os sintomas do diabetes tipo 2?"

# Via API (mesmo protocolo OpenAI)
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:4b",
    "messages": [{"role": "user", "content": "Quais são os sintomas do diabetes?"}]
  }'
```

---

## MODO 3 — Ollama + Phi-3-mini fine-tunado (phi3-medassist)

Usa o **seu próprio modelo treinado no Colab**, convertido para GGUF e registrado no Ollama.
Máxima especialização médica — roda sem GPU, sem API key, offline.

### Passo 2 — Baixar o GGUF do Google Drive

O arquivo `phi-3-mini-4k-instruct.Q4_K_M.gguf` (~2.2 GB) foi gerado no Colab
(Células 12b/12c do notebook) e está disponível via Google Drive:

> 📁 **Link do Google Drive:** `[link será disponibilizado pelo autor]`

Após baixar, mova para a pasta correta:

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

### Passo 3 — Criar o Modelfile

**Mac/Linux:**
```bash
cat > outputs/gguf/Modelfile << 'MODELEOF'
FROM ./phi3-medassist.gguf

SYSTEM "Você é um assistente médico especializado. Responda de forma precisa, clara e em português. Sempre recomende consulta com profissional de saúde para diagnósticos e tratamentos."

PARAMETER temperature 0.3
PARAMETER num_ctx 2048
MODELEOF
```

**Windows (PowerShell):**
```powershell
@"
FROM ./phi3-medassist.gguf

SYSTEM "Você é um assistente médico especializado. Responda de forma precisa, clara e em português. Sempre recomende consulta com profissional de saúde para diagnósticos e tratamentos."

PARAMETER temperature 0.3
PARAMETER num_ctx 2048
"@ | Out-File -Encoding utf8 outputs\gguf\Modelfile
```

### Passo 4 — Registrar no Ollama

```bash
cd outputs/gguf
ollama create phi3-medassist -f Modelfile
# Aguarde 1-2 minutos

# Verificar
ollama list
# Deve aparecer: phi3-medassist
```

### Passo 5 — Configurar o .env

```env
LLM_BACKEND=openai
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=phi3-medassist
```

### Testar o MODO 3

```bash
ollama run phi3-medassist "Quais são os sintomas do diabetes tipo 2?"
```

---

## Configurar pelo modal da interface web (qualquer modo)

1. Abrir `http://localhost:8080`
2. Clicar em **"Configurar Backend"** no header
3. Selecionar o modo desejado:
   - **Ollama — Gemma 3 local** → MODO 2
   - **Phi-3-mini fine-tunado** → MODO 3
4. Preencher URL e modelo (já preenchidos por padrão)
5. Clicar em **"Aplicar configuração"**

Nenhum reinício necessário — o backend troca em tempo real.

---

## Velocidade esperada

| Hardware | MODO 2 (gemma3:4b) | MODO 3 (phi3-medassist) |
|---|---|---|
| CPU Intel/AMD 8 cores, 16GB RAM | ~20-45s | ~25-50s |
| CPU Intel/AMD 8 cores, 32GB RAM | ~10-25s | ~15-30s |
| GPU NVIDIA 4GB+ VRAM | ~3-8s | ~4-10s |
| GPU NVIDIA 8GB+ VRAM | ~2-5s | ~3-7s |

O Ollama detecta GPU automaticamente se disponível (NVIDIA CUDA ou Apple Metal).

---

## Comandos úteis do Ollama

```bash
# Ver modelos registrados
ollama list

# Remover modelo para liberar espaço
ollama rm gemma3:4b
ollama rm phi3-medassist

# Ver logs do servidor
ollama logs

# Verificar status do servidor
curl http://localhost:11434/api/version
```

---

## Como o GGUF foi gerado (para referência)

O arquivo `phi-3-mini-4k-instruct.Q4_K_M.gguf` foi criado no Google Colab
executando as **Células 12b e 12c** do notebook `MedAssist_FineTuning_Colab.ipynb`:

1. **Célula 12b** — funde os adaptadores LoRA ao Phi-3-mini base e exporta em formato GGUF quantizado (q4_k_m)
2. **Célula 12c** — faz o download do arquivo para sua máquina

Se quiser gerar seu próprio GGUF a partir dos adaptadores LoRA:
1. Abra o notebook no Colab com GPU T4 ativa
2. Execute apenas as Células 12b e 12c (não precisa treinar novamente)
3. Tempo estimado: ~5-10 minutos

---

## Referências

- [Ollama — Download](https://ollama.com/download)
- [Gemma 3 no Ollama](https://ollama.com/library/gemma3)
- [Gemma 3 — Google DeepMind](https://deepmind.google/models/gemma/)
- [Phi-3-mini — Microsoft](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/)
