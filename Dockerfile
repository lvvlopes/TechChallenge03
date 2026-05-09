FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

# Instala PyTorch CPU primeiro (versão fixada para evitar conflitos)
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Demais dependências Python
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.0 \
    "transformers>=4.40.0,<4.46.0" \
    "peft>=0.10.0" \
    "accelerate>=0.30.0" \
    sentencepiece \
    protobuf

# Código e adaptadores LoRA do Colab
COPY local_server.py .
COPY config.py .
COPY utils.py .
COPY outputs/model/ ./outputs/model/

EXPOSE 8000

# Health check — aguarda até 5 min (tempo de carregar o Phi-3-mini)
HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=10 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "local_server.py"]
