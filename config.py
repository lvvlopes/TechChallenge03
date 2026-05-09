"""
config.py - Configuracoes globais do MedAssist
LLM: OpenAI API (gpt-4o-mini por padrao)
Fine-tuning de referencia: Phi-3-mini (adaptadores em outputs/model/)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# ─────────────────────────────────────────────
# Caminhos do projeto
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
LOGS_DIR    = BASE_DIR / "logs"
OUTPUTS_DIR = BASE_DIR / "outputs"

DATASET_PATH = DATA_DIR / "medquad_ptbr.jsonl"
TRAIN_PATH   = DATA_DIR / "train.jsonl"
VAL_PATH     = DATA_DIR / "val.jsonl"

MODEL_OUTPUT_DIR   = OUTPUTS_DIR / "model"   # adaptadores LoRA do Colab
LORA_ADAPTERS_PATH = OUTPUTS_DIR / "model"

# ─────────────────────────────────────────────
# Backend de LLM — escolha aqui qual usar
# ─────────────────────────────────────────────
# "openai"  -> OpenAI API (gpt-4o-mini). Requer OPENAI_API_KEY no .env.
#              Rapido, sem GPU, ideal para producao.
# "local"   -> Phi-3-mini + adaptadores LoRA gerados no Colab.
#              Nao requer API key. Mais lento em CPU, exige ~8GB RAM.
#              Demonstra o fine-tuning real realizado no projeto.
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")   # "openai" | "local"

# ─────────────────────────────────────────────
# OpenAI API — LLM principal
# ─────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
# OPENAI_BASE_URL:
#   vazio          -> OpenAI oficial (https://api.openai.com)
#   localhost:8000 -> local_server.py (Phi-3 + LoRA na sua maquina)
#   https://...    -> Azure Container Apps (Phi-3 + LoRA na nuvem)

# ─────────────────────────────────────────────
# Modelo fine-tuned (Phi-3-mini + LoRA Colab)
# ─────────────────────────────────────────────
BASE_MODEL_NAME    = "microsoft/Phi-3-mini-4k-instruct"
UNSLOTH_MODEL_NAME = "unsloth/Phi-3-mini-4k-instruct"
LOCAL_MODEL_MAX_NEW_TOKENS = 512
LOCAL_MODEL_DEVICE = "auto"        # "auto" detecta GPU; "cpu" forca CPU

# ─────────────────────────────────────────────
# LoRA (valores usados no Colab)
# ─────────────────────────────────────────────
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

TRAIN_EPOCHS     = 1
TRAIN_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE    = 2e-4
WARMUP_RATIO     = 0.03
MAX_SEQ_LENGTH   = 1024
WEIGHT_DECAY     = 0.01
VAL_SPLIT_RATIO  = 0.1
SAVE_STEPS       = 200
LOGGING_STEPS    = 25

# ─────────────────────────────────────────────
# Assistente / RAG
# ─────────────────────────────────────────────
EMBEDDING_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_STORE_PATH = OUTPUTS_DIR / "vectorstore"
TOP_K_RETRIEVAL   = 3
MAX_NEW_TOKENS    = 512
TEMPERATURE       = 0.3

SYSTEM_PROMPT = (
    "Voce e um assistente medico especializado, treinado com dados do MedQuAD em portugues. "
    "Responda de forma precisa, clara e segura em portugues brasileiro, "
    "sempre recomendando consulta medica profissional para decisoes clinicas. "
    "Cite sempre a fonte da informacao utilizada na resposta."
)

# ─────────────────────────────────────────────
# Seguranca
# ─────────────────────────────────────────────
SAFETY_DISCLAIMER = (
    "\n\nAviso Medico: Esta resposta e gerada por IA e tem carater informativo. "
    "Nao substitui a avaliacao de um profissional de saude habilitado. "
    "Consulte sempre um medico antes de tomar qualquer decisao clinica."
)

EMERGENCY_KEYWORDS = [
    "infarto", "derrame", "avc", "parada cardiaca", "nao respira",
    "inconsciente", "sangramento intenso", "convulsao", "overdose",
    "suicidio", "me matar", "emergencia", "socorro",
]

PRESCRIPTION_KEYWORDS = [
    "prescreva", "receita", "tome", "usar", "tomar", "dosagem",
    "mg por dia", "comprimido",
]

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
AUDIT_LOG_PATH    = LOGS_DIR / "audit.log"
TRAINING_LOG_PATH = LOGS_DIR / "training.log"
LOG_LEVEL         = "INFO"

# ─────────────────────────────────────────────
# Criar diretorios
# ─────────────────────────────────────────────
for _d in [DATA_DIR, LOGS_DIR, OUTPUTS_DIR, MODEL_OUTPUT_DIR,
           VECTOR_STORE_PATH, DATA_DIR / "processed"]:
    _d.mkdir(parents=True, exist_ok=True)
