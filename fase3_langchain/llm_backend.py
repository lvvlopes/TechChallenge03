"""
fase3_langchain/llm_backend.py

Backend unificado de LLM para o MedAssist.
Suporta 2 modos configurados via .env:

  MODO 1: LLM_BACKEND=openai  + OPENAI_API_KEY          → OpenAI API (gpt-4o-mini)
  MODO 2: LLM_BACKEND=openai  + OPENAI_BASE_URL=http://localhost:11434/v1 → Ollama (gemma3:4b)
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
# Backend OpenAI / Ollama (protocolo OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_openai(question: str, context: str) -> str:
    """
    Gera resposta via:
      - MODO 1: OpenAI API oficial      (OPENAI_BASE_URL vazio)
      - MODO 2: Ollama gemma3:4b local  (OPENAI_BASE_URL=http://localhost:11434/v1)
    Ambos usam o mesmo protocolo OpenAI-compatible.
    """
    effective_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    effective_url = OPENAI_BASE_URL or os.getenv("OPENAI_BASE_URL", "")

    if not effective_key and not effective_url:
        raise RuntimeError(
            "OPENAI_API_KEY nao encontrada no .env.\n"
            "Configure o MODO 1 (OpenAI) ou MODO 2 (Ollama) corretamente."
        )

    from openai import OpenAI

    client_kwargs = {"api_key": effective_key or "ollama"}
    if effective_url:
        client_kwargs["base_url"] = effective_url

    # Timeout maior para Ollama em CPU (gemma3:4b pode levar ~30-60s)
    client_kwargs["timeout"] = 300.0

    client = OpenAI(**client_kwargs)

    model_to_use = os.getenv("OPENAI_MODEL", OPENAI_MODEL)

    if context:
        user_msg = (
            f"Pergunta do usuário: {question}\n\n"
            f"---\n"
            f"Informações de referência extraídas da base MedQuAD "
            f"(use apenas como contexto de apoio, responda à pergunta do usuário acima):\n\n"
            f"{context}\n"
            f"---\n\n"
            f"Responda à pergunta do usuário em português, de forma clara e objetiva."
        )
    else:
        user_msg = (
            f"Pergunta do usuário: {question}\n\n"
            f"Não encontrei informações específicas na base MedQuAD. "
            f"Responda com seu conhecimento médico geral e recomende consulta profissional."
        )

    response = client.chat.completions.create(
        model=model_to_use,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=512,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Backend Local direto em RAM (requer GPU — para uso com local_server.py)
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
            "  ERRO: backend local requer GPU\n"
            "=" * 60 + "\n\n"
            "  O Phi-3-mini (3.8B params) precisa de ~8GB VRAM.\n\n"
            "  Use um dos modos suportados:\n\n"
            "  MODO 1 — OpenAI API:\n"
            "       LLM_BACKEND=openai\n"
            "       OPENAI_API_KEY=sk-...\n\n"
            "  MODO 2 — Ollama + Gemma 3 local:\n"
            "       LLM_BACKEND=openai\n"
            "       OPENAI_BASE_URL=http://localhost:11434/v1\n"
            "       OPENAI_API_KEY=ollama\n"
            "       OPENAI_MODEL=gemma3:4b\n"
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
        f"Pergunta do usuário: {question}\n\n"
        f"---\n"
        f"Informações de referência (base MedQuAD):\n{context}\n"
        f"---\n\n"
        f"Responda à pergunta do usuário em português."
        if context else
        f"Pergunta do usuário: {question}\n\n"
        f"Responda com conhecimento médico geral em português."
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
        if base_url and "11434" in base_url:
            desc = f"Ollama — Gemma 3 local (gemma3:4b) via {base_url}"
            mode = "ollama"
        elif base_url and "localhost" in base_url:
            desc = f"Servidor Local Phi-3-mini + LoRA ({base_url})"
            mode = "local-server"
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
