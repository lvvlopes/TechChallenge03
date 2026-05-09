"""
local_server.py — Servidor local OpenAI-compatible para o MedAssist

Carrega o Phi-3-mini + adaptadores LoRA (gerados no Colab) UMA vez na memória
e expõe o endpoint POST /v1/chat/completions — idêntico à OpenAI API.

O assistente consome normalmente via llm_backend.py sem saber que é local.

Uso:
    # Terminal 1 — deixar rodando
    python local_server.py

    # Terminal 2 — configurar e rodar o assistente
    # No .env:
    #   LLM_BACKEND=openai
    #   OPENAI_API_KEY=local        (qualquer valor, não é validado)
    #   OPENAI_BASE_URL=http://localhost:8000/v1
    python fase3_langchain/assistant.py

Requisitos:
    pip install fastapi uvicorn transformers peft accelerate
    # Com GPU CUDA:
    pip install bitsandbytes
"""

import sys
import time
import logging
from pathlib import Path

# ── Logging básico ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("medassist-server")

# ── Caminhos ──────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).parent
ADAPTERS_PATH    = ROOT / "outputs" / "model"
BASE_MODEL_NAME  = "microsoft/Phi-3-mini-4k-instruct"
HOST             = "0.0.0.0"
PORT             = 8000
MAX_NEW_TOKENS   = 512


# ── Verificação dos adaptadores ───────────────────────────────────────────────
def check_adapters():
    required = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [f for f in required if not (ADAPTERS_PATH / f).exists()]
    if missing:
        log.error("Adaptadores LoRA não encontrados em: %s", ADAPTERS_PATH)
        log.error("Arquivos faltando: %s", missing)
        log.error("Execute o notebook MedAssist_FineTuning_Colab.ipynb no Colab")
        log.error("e copie os arquivos gerados para outputs/model/")
        sys.exit(1)
    log.info("Adaptadores encontrados em: %s", ADAPTERS_PATH)


# ── Carregamento do modelo ────────────────────────────────────────────────────
_model     = None
_tokenizer = None


def load_model():
    global _model, _tokenizer

    log.info("Carregando tokenizer de: %s", ADAPTERS_PATH)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    _tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTERS_PATH),
        trust_remote_code=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Dispositivo: %s", device.upper())

    log.info("Carregando modelo base: %s", BASE_MODEL_NAME)
    log.info("(pode levar alguns minutos na primeira vez...)")

    if device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            )
            log.info("Modelo carregado com quantização 4-bit (GPU)")
        except ImportError:
            log.warning("bitsandbytes não instalado — carregando em float16 na GPU")
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            )
    else:
        log.warning("GPU não detectada — carregando em CPU (respostas mais lentas)")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

    log.info("Aplicando adaptadores LoRA de: %s", ADAPTERS_PATH)
    _model = PeftModel.from_pretrained(base, str(ADAPTERS_PATH))
    _model.eval()
    log.info("Modelo pronto! Servidor iniciando em http://localhost:%d", PORT)


# ── Inferência ────────────────────────────────────────────────────────────────
def generate(messages: list[dict], max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.7) -> str:
    import torch

    input_ids = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    device = next(_model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output_ids = _model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── API FastAPI ───────────────────────────────────────────────────────────────
def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        log.error("FastAPI não instalado. Execute: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(title="MedAssist Local Server", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Modelos de request/response (compatível com OpenAI) ──────────────────
    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = "phi3-medassist"
        messages: list[Message]
        max_tokens: int = MAX_NEW_TOKENS
        temperature: float = 0.7

    class Choice(BaseModel):
        index: int = 0
        message: Message
        finish_reason: str = "stop"

    class Usage(BaseModel):
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0

    class ChatResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: list[Choice]
        usage: Usage

    # ── Endpoints ─────────────────────────────────────────────────────────────
    @app.get("/")
    def root():
        return {
            "status": "online",
            "model": "phi3-medassist",
            "adapters": str(ADAPTERS_PATH),
            "docs": f"http://localhost:{PORT}/docs",
        }

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{
                "id": "phi3-medassist",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "medassist-local",
            }]
        }

    @app.post("/v1/chat/completions", response_model=ChatResponse)
    def chat_completions(req: ChatRequest):
        t0 = time.time()
        messages = [{"role": m.role, "content": m.content} for m in req.messages]

        log.info(
            "Gerando resposta | msgs=%d | max_tokens=%d | temp=%.1f",
            len(messages), req.max_tokens, req.temperature,
        )

        answer = generate(
            messages=messages,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
        )

        elapsed = time.time() - t0
        log.info("Resposta gerada em %.1fs (%d chars)", elapsed, len(answer))

        return ChatResponse(
            id=f"chatcmpl-local-{int(t0)}",
            created=int(t0),
            model=req.model,
            choices=[Choice(message=Message(role="assistant", content=answer))],
            usage=Usage(completion_tokens=len(answer.split())),
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": _model is not None}

    return app


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        log.error("uvicorn não instalado. Execute: pip install fastapi uvicorn")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  MedAssist — Servidor Local Phi-3-mini + LoRA")
    print("=" * 60)
    print(f"  Adaptadores : {ADAPTERS_PATH}")
    print(f"  Endpoint    : http://localhost:{PORT}/v1/chat/completions")
    print(f"  Docs        : http://localhost:{PORT}/docs")
    print("=" * 60)
    print()

    check_adapters()
    load_model()

    app = create_app()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
