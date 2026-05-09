"""
fase3_langchain/llm_backend.py

Backend unificado de LLM para o MedAssist.
Suporta 3 modos configurados via .env:

  LLM_BACKEND=openai  + OPENAI_API_KEY          → OpenAI API (gpt-4o-mini)
  LLM_BACKEND=openai  + OPENAI_BASE_URL=http://localhost:8000/v1  → local_server.py
  LLM_BACKEND=openai  + OPENAI_BASE_URL=https://<azure-url>/v1   → Azure Container Apps
  LLM_BACKEND=local                              → Phi-3-mini + LoRA direto em RAM (requer GPU)
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    LLM_BACKEND,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL,
    LORA_ADAPTERS_PATH, BASE_MODEL_NAME,
    SYSTEM_PROMPT, TEMPERATURE,
    LOCAL_MODEL_MAX_NEW_TOKENS, LOCAL_MODEL_DEVICE,
)

_local_model     = None
_local_tokenizer = None


# ─────────────────────────────────────────────────────────────────────────────
# Backend OpenAI / Servidor compatível (local ou Azure)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_openai(question: str, context: str) -> str:
    """
    Gera resposta via:
      - OpenAI API oficial (OPENAI_BASE_URL vazio)
      - local_server.py   (OPENAI_BASE_URL=http://localhost:8000/v1)
      - Azure             (OPENAI_BASE_URL=https://<app>.azurecontainerapps.io/v1)
    Todos usam o mesmo protocolo OpenAI-compatible.
    """
    effective_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    effective_url = OPENAI_BASE_URL or os.getenv("OPENAI_BASE_URL", "")

    # Servidores locais/Azure nao precisam de key real
    if not effective_key and not effective_url:
        raise RuntimeError(
            "OPENAI_API_KEY nao encontrada no .env.\n"
            "Configure LLM_BACKEND e credenciais corretamente."
        )

    from openai import OpenAI

    client_kwargs = {"api_key": effective_key or "local-or-azure"}
    if effective_url:
        client_kwargs["base_url"] = effective_url

    # Timeout alto para Azure/local em CPU (Phi-3-mini leva 2-5 min por resposta)
    client_kwargs["timeout"] = 600.0  # 10 minutos

    client = OpenAI(**client_kwargs)

    model_to_use = os.getenv("OPENAI_MODEL", OPENAI_MODEL)

    # Reduz tokens para Azure CPU (menos tokens = resposta mais rapida)
    is_azure_or_local = bool(effective_url)
    max_tok = 256 if is_azure_or_local else 512

    if context:
        user_msg = (
            f"Use o contexto abaixo (extraido do MedQuAD) para responder a pergunta.\n\n"
            f"Contexto:\n{context}\n\n"
            f"Pergunta: {question}"
        )
    else:
        user_msg = (
            f"Nao encontrei informacoes especificas na base MedQuAD. "
            f"Responda com seu conhecimento medico geral e recomende consulta profissional.\n\n"
            f"Pergunta: {question}"
        )

    response = client.chat.completions.create(
        model=model_to_use,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=max_tok,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Backend Local direto em RAM (requer GPU)
# ─────────────────────────────────────────────────────────────────────────────

def _load_local_model():
    global _local_model, _local_tokenizer

    if _local_model is not None:
        return _local_model, _local_tokenizer

    adapters_path = str(LORA_ADAPTERS_PATH)
    if not Path(adapters_path).exists():
        raise RuntimeError(
            f"Adaptadores LoRA nao encontrados em: {adapters_path}\n"
            "Execute o notebook MedAssist_FineTuning_Colab.ipynb no Google Colab "
            "e copie os arquivos gerados para outputs/model/."
        )

    import torch

    if LOCAL_MODEL_DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = LOCAL_MODEL_DEVICE

    if device == "cpu":
        raise RuntimeError(
            "\n" + "=" * 60 + "\n"
            "  ERRO: LLM_BACKEND=local requer GPU\n"
            "=" * 60 + "\n\n"
            "  O Phi-3-mini (3.8B params) precisa de ~8GB VRAM.\n"
            "  Sem GPU, travaria a maquina durante o carregamento.\n\n"
            "  Solucoes recomendadas:\n\n"
            "  1. OpenAI API (mais simples):\n"
            "       LLM_BACKEND=openai\n"
            "       OPENAI_API_KEY=sk-...\n\n"
            "  2. Servidor local com GPU (local_server.py):\n"
            "       python local_server.py\n"
            "       LLM_BACKEND=openai\n"
            "       OPENAI_BASE_URL=http://localhost:8000/v1\n"
            "       OPENAI_API_KEY=local\n\n"
            "  3. Azure Container Apps (nuvem, sem GPU local):\n"
            "       LLM_BACKEND=openai\n"
            "       OPENAI_BASE_URL=https://<app>.azurecontainerapps.io/v1\n"
            "       OPENAI_API_KEY=azure\n"
            + "=" * 60
        )

    print(f"[LOCAL] Modelo base: {BASE_MODEL_NAME}")
    print(f"[LOCAL] Adaptadores: {adapters_path}")
    print(f"[LOCAL] Dispositivo: {device.upper()}")
    print("[LOCAL] Carregando... (pode levar alguns minutos)")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(adapters_path, trust_remote_code=True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        print("[LOCAL] Aplicando adaptadores LoRA do Colab...")
        model = PeftModel.from_pretrained(base_model, adapters_path)
        model.eval()

        _local_model     = model
        _local_tokenizer = tokenizer
        print("[LOCAL] Modelo carregado com sucesso!")
        return _local_model, _local_tokenizer

    except ImportError as e:
        raise RuntimeError(
            f"Dependencia faltando: {e}\n"
            "Instale: pip install transformers peft accelerate bitsandbytes"
        )


def _generate_local(question: str, context: str) -> str:
    import torch
    model, tokenizer = _load_local_model()

    user_content = (
        f"Use o contexto abaixo para responder.\n\nContexto:\n{context}\n\nPergunta: {question}"
        if context else
        f"Responda com conhecimento medico geral.\n\nPergunta: {question}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=LOCAL_MODEL_MAX_NEW_TOKENS,
            temperature=TEMPERATURE if TEMPERATURE > 0 else None,
            do_sample=TEMPERATURE > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Interface pública
# ─────────────────────────────────────────────────────────────────────────────

def generate(question: str, context: str = "") -> str:
    backend = LLM_BACKEND.lower()
    if backend == "openai":
        return _generate_openai(question, context)
    elif backend == "local":
        return _generate_local(question, context)
    else:
        raise ValueError(f"LLM_BACKEND invalido: '{LLM_BACKEND}'. Use 'openai' ou 'local'.")


def get_backend_info() -> dict:
    backend  = LLM_BACKEND.lower()
    base_url = OPENAI_BASE_URL or os.getenv("OPENAI_BASE_URL", "")

    if backend == "openai":
        if base_url and "localhost" in base_url:
            desc = f"Servidor Local ({base_url})"
            mode = "local-server"
        elif base_url and ("azure" in base_url or "https" in base_url):
            desc = f"Azure Container Apps ({base_url})"
            mode = "azure"
        else:
            desc = f"OpenAI API ({OPENAI_MODEL})"
            mode = "openai"

        return {
            "backend":     "openai",
            "mode":        mode,
            "model":       OPENAI_MODEL,
            "base_url":    base_url or "https://api.openai.com",
            "description": desc + " + RAG MedQuAD",
            "api_key_set": bool(OPENAI_API_KEY or base_url),
        }

    elif backend == "local":
        import torch
        has_gpu = torch.cuda.is_available() if _is_torch_available() else False
        return {
            "backend":       "local",
            "mode":          "local-ram",
            "model":         BASE_MODEL_NAME,
            "adapters":      str(LORA_ADAPTERS_PATH),
            "description":   "Phi-3-mini + LoRA (Colab) em RAM + RAG MedQuAD",
            "gpu_available": has_gpu,
            "adapters_exist": Path(LORA_ADAPTERS_PATH).exists(),
        }

    return {"backend": "desconhecido", "model": LLM_BACKEND}


def _is_torch_available() -> bool:
    try:
        import torch; return True  # noqa
    except ImportError:
        return False
