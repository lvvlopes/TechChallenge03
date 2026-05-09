# Guia de Setup — Ollama + Gemma 3 (gemma3:4b)
## MedAssist · Backend local sem GPU obrigatória

O Ollama permite rodar o **Gemma 3 (4B parâmetros)** localmente com uma API idêntica à OpenAI.
Sem API key, sem custo, sem dependência de nuvem — o modelo utilizado neste projeto.

---

## Por que Ollama + Gemma 3?

| | OpenAI API | Gemma 3 via Ollama |
|---|---|---|
| GPU obrigatória | ❌ | ❌ |
| API key | ✅ (paga) | ❌ |
| Velocidade | ~2s | ~5-30s (depende do hardware) |
| Funciona offline | ❌ | ✅ |
| Custo | Por token | Gratuito |

> O **gemma3:4b** é o modelo Ollama configurado neste projeto para o MODO 2.
> Roda em CPU com ~8GB de RAM ou em GPU NVIDIA/AMD com melhor performance.

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

## Passo 2 — Baixar o Gemma 3 (4B)

```bash
# Modelo usado neste projeto — ~3.3 GB, 4B parâmetros
ollama pull gemma3:4b
```

O download acontece uma vez e fica salvo localmente (`~/.ollama/models/`).

---

## Passo 3 — Iniciar o servidor Ollama

```bash
# Windows/Mac: o Ollama inicia automaticamente após instalação.
# Linux: inicie manualmente:
ollama serve
```

Verificar se está rodando:
```bash
curl http://localhost:11434/api/tags
# Deve retornar a lista de modelos baixados, incluindo gemma3:4b
```

---

## Passo 4 — Configurar o MedAssist (.env)

No arquivo `.env`, ative o MODO 2:

```env
# Comente o MODO 1 (OpenAI) e ative o MODO 2 (Ollama):
# LLM_BACKEND=openai
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=

LLM_BACKEND=openai
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=gemma3:4b
```

Rodar o assistente:
```bash
python web_app.py
# Acesse: http://localhost:8080
```

---

## Ou configurar pelo modal da interface web

1. Abrir http://localhost:8080
2. Clicar em **"Configurar Backend"** no header
3. Selecionar **Ollama — Gemma 3 local**
4. URL: `http://localhost:11434/v1`
5. Modelo: `gemma3:4b`
6. Clicar em **"Aplicar configuração"**

Nenhum reinício necessário — o backend troca em tempo real.

---

## Testar diretamente

```bash
# Testar via CLI do Ollama
ollama run gemma3:4b "Quais são os sintomas do diabetes tipo 2?"

# Testar via API (mesmo protocolo OpenAI)
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:4b",
    "messages": [{"role": "user", "content": "Quais são os sintomas do diabetes?"}]
  }'
```

---

## Velocidade esperada com gemma3:4b

| Hardware | Tempo por resposta |
|---|---|
| CPU Intel/AMD 8 cores, 16GB RAM | ~20-45s |
| CPU Intel/AMD 8 cores, 32GB RAM | ~10-25s |
| GPU NVIDIA 4GB+ VRAM | ~3-8s |
| GPU NVIDIA 8GB+ VRAM | ~2-5s |

O Ollama detecta GPU automaticamente se disponível (NVIDIA CUDA ou Apple Metal).

---

## Comandos úteis do Ollama

```bash
# Ver modelos baixados
ollama list

# Remover modelo para liberar espaço
ollama rm gemma3:4b

# Ver logs do servidor
ollama logs

# Verificar status do servidor
curl http://localhost:11434/api/version
```

---

## Referências

- [Ollama — Download](https://ollama.com/download)
- [Gemma 3 no Ollama](https://ollama.com/library/gemma3)
- [Gemma 3 — Google DeepMind](https://deepmind.google/models/gemma/)
