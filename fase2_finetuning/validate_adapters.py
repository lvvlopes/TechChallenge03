"""
fase2_finetuning/validate_adapters.py
Valida os adaptadores LoRA gerados no Google Colab.

Verifica:
  1. Presença e integridade dos arquivos obrigatórios
  2. Conteúdo e coerência do adapter_config.json
  3. Estrutura e integridade do adapter_model.safetensors
  4. Tokenizer funcional
  5. Chat template presente e válido
  6. Relatório completo com aprovação/reprovação por checagem

Execute: python fase2_finetuning/validate_adapters.py
"""
import sys
import io
import os
import json
import struct
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_OUTPUT_DIR

# ─────────────────────────────────────────────
# Configurações esperadas do fine-tuning
# ─────────────────────────────────────────────
EXPECTED = {
    "peft_type":   "LORA",
    "task_type":   "CAUSAL_LM",
    "lora_r":      16,
    "lora_alpha":  32,
    "target_modules": {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    },
    # Tamanho mínimo esperado para os pesos LoRA (Phi-3-mini, rank=16)
    "min_safetensors_mb": 50,
    # Número mínimo de tensores LoRA esperados
    "min_tensors": 100,
    # Prefixo esperado nos nomes dos tensores
    "tensor_prefix": "base_model.model",
    # Sufixos LoRA esperados
    "lora_suffixes": {"lora_A.weight", "lora_B.weight"},
}

REQUIRED_FILES = {
    "adapter_config.json":      "Configuração LoRA",
    "adapter_model.safetensors": "Pesos dos adaptadores",
    "tokenizer.json":            "Vocabulário do tokenizer",
    "tokenizer_config.json":     "Configuração do tokenizer",
}

OPTIONAL_FILES = {
    "tokenizer.model":     "Modelo sentencepiece (opcional)",
    "chat_template.jinja": "Template de chat (opcional)",
    "README.md":           "Documentação (opcional)",
}


# ─────────────────────────────────────────────
# Utilitários de relatório
# ─────────────────────────────────────────────
class Report:
    def __init__(self):
        self.checks: list[dict] = []

    def ok(self, section: str, msg: str, detail: str = ""):
        self.checks.append({"status": "OK", "section": section, "msg": msg, "detail": detail})
        detail_str = f" ({detail})" if detail else ""
        print(f"  [OK]  {msg}{detail_str}")

    def warn(self, section: str, msg: str, detail: str = ""):
        self.checks.append({"status": "WARN", "section": section, "msg": msg, "detail": detail})
        detail_str = f" ({detail})" if detail else ""
        print(f"  [AVISO] {msg}{detail_str}")

    def fail(self, section: str, msg: str, detail: str = ""):
        self.checks.append({"status": "FAIL", "section": section, "msg": msg, "detail": detail})
        detail_str = f" ({detail})" if detail else ""
        print(f"  [FALHA] {msg}{detail_str}")

    def summary(self) -> dict:
        total = len(self.checks)
        ok    = sum(1 for c in self.checks if c["status"] == "OK")
        warns = sum(1 for c in self.checks if c["status"] == "WARN")
        fails = sum(1 for c in self.checks if c["status"] == "FAIL")
        return {"total": total, "ok": ok, "warnings": warns, "failures": fails}

    def approved(self) -> bool:
        return all(c["status"] != "FAIL" for c in self.checks)


# ─────────────────────────────────────────────
# Checagens
# ─────────────────────────────────────────────

def check_files(path: Path, report: Report):
    """1. Verifica presença e tamanho mínimo dos arquivos."""
    print("\n[1] Arquivos")

    for fname, desc in REQUIRED_FILES.items():
        fpath = path / fname
        if not fpath.exists():
            report.fail("arquivos", f"Arquivo obrigatorio ausente: {fname}", desc)
        else:
            size_mb = fpath.stat().st_size / 1024**2
            report.ok("arquivos", f"{fname}", f"{size_mb:.1f} MB")

    for fname, desc in OPTIONAL_FILES.items():
        fpath = path / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1024**2
            report.ok("arquivos", f"{fname} (opcional)", f"{size_mb:.1f} MB")
        else:
            report.warn("arquivos", f"{fname} ausente (opcional)", desc)


def check_adapter_config(path: Path, report: Report) -> dict:
    """2. Valida o conteúdo do adapter_config.json."""
    print("\n[2] adapter_config.json")
    cfg_path = path / "adapter_config.json"
    if not cfg_path.exists():
        report.fail("config", "adapter_config.json nao encontrado")
        return {}

    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        report.fail("config", "Falha ao ler adapter_config.json", str(e))
        return {}

    # peft_type
    peft_type = cfg.get("peft_type", "")
    if peft_type == EXPECTED["peft_type"]:
        report.ok("config", "peft_type", peft_type)
    else:
        report.fail("config", f"peft_type incorreto: '{peft_type}'", f"esperado: {EXPECTED['peft_type']}")

    # task_type
    task_type = cfg.get("task_type", "")
    if task_type == EXPECTED["task_type"]:
        report.ok("config", "task_type", task_type)
    else:
        report.fail("config", f"task_type incorreto: '{task_type}'", f"esperado: {EXPECTED['task_type']}")

    # LoRA rank
    r = cfg.get("r", 0)
    if r == EXPECTED["lora_r"]:
        report.ok("config", f"lora rank (r)", str(r))
    else:
        report.warn("config", f"lora rank (r) = {r}", f"esperado: {EXPECTED['lora_r']}")

    # LoRA alpha
    alpha = cfg.get("lora_alpha", 0)
    if alpha == EXPECTED["lora_alpha"]:
        report.ok("config", f"lora_alpha", str(alpha))
    else:
        report.warn("config", f"lora_alpha = {alpha}", f"esperado: {EXPECTED['lora_alpha']}")

    # target_modules
    target = set(cfg.get("target_modules", []))
    expected_target = EXPECTED["target_modules"]
    missing_modules = expected_target - target
    extra_modules   = target - expected_target
    if not missing_modules:
        report.ok("config", "target_modules", ", ".join(sorted(target)))
    else:
        report.warn("config", f"target_modules incompleto", f"faltando: {missing_modules}")
    if extra_modules:
        report.warn("config", f"target_modules extras detectados", str(extra_modules))

    # inference_mode
    if cfg.get("inference_mode") is True:
        report.ok("config", "inference_mode = True (pronto para uso)")
    else:
        report.warn("config", "inference_mode != True", "verifique se o modelo foi salvo corretamente")

    # base_model
    base = cfg.get("base_model_name_or_path", "")
    if "phi" in base.lower() or "microsoft" in base.lower():
        report.ok("config", "base_model_name_or_path", base)
    elif base:
        report.warn("config", f"base_model_name_or_path inesperado: {base}")
    else:
        report.fail("config", "base_model_name_or_path ausente no config")

    return cfg


def check_safetensors(path: Path, report: Report):
    """3. Valida integridade e estrutura do adapter_model.safetensors."""
    print("\n[3] adapter_model.safetensors")
    st_path = path / "adapter_model.safetensors"
    if not st_path.exists():
        report.fail("safetensors", "Arquivo nao encontrado")
        return

    size_mb = st_path.stat().st_size / 1024**2

    # Tamanho mínimo
    if size_mb >= EXPECTED["min_safetensors_mb"]:
        report.ok("safetensors", f"Tamanho do arquivo", f"{size_mb:.1f} MB")
    else:
        report.fail(
            "safetensors",
            f"Arquivo muito pequeno: {size_mb:.1f} MB",
            f"minimo esperado: {EXPECTED['min_safetensors_mb']} MB — fine-tuning pode ter sido interrompido",
        )

    # Lê header do safetensors
    try:
        with open(st_path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            if header_len > 100 * 1024 * 1024:
                report.fail("safetensors", "Header corrompido (tamanho invalido)")
                return
            header_bytes = f.read(header_len)
            meta = json.loads(header_bytes.decode("utf-8"))
    except Exception as e:
        report.fail("safetensors", "Falha ao ler header do safetensors", str(e))
        return

    tensors = [k for k in meta.keys() if k != "__metadata__"]
    n = len(tensors)

    # Quantidade de tensores
    if n >= EXPECTED["min_tensors"]:
        report.ok("safetensors", f"Numero de tensores LoRA", str(n))
    else:
        report.fail(
            "safetensors",
            f"Poucos tensores: {n}",
            f"minimo esperado: {EXPECTED['min_tensors']} — fine-tuning incompleto?",
        )

    # Prefixo esperado
    wrong_prefix = [k for k in tensors if not k.startswith(EXPECTED["tensor_prefix"])]
    if not wrong_prefix:
        report.ok("safetensors", f"Prefixo dos tensores correto", EXPECTED["tensor_prefix"])
    else:
        report.warn("safetensors", f"{len(wrong_prefix)} tensores com prefixo inesperado", str(wrong_prefix[:3]))

    # Sufixos lora_A / lora_B
    lora_a = [k for k in tensors if k.endswith("lora_A.weight")]
    lora_b = [k for k in tensors if k.endswith("lora_B.weight")]
    if lora_a and lora_b and len(lora_a) == len(lora_b):
        report.ok("safetensors", f"Pares lora_A/lora_B balanceados", f"{len(lora_a)} pares")
    else:
        report.fail(
            "safetensors",
            f"Pares lora_A/lora_B desbalanceados",
            f"lora_A: {len(lora_a)} | lora_B: {len(lora_b)}",
        )

    # Módulos treinados
    modules_found = set()
    for k in tensors:
        for mod in EXPECTED["target_modules"]:
            if f".{mod}." in k:
                modules_found.add(mod)
    missing_mods = EXPECTED["target_modules"] - modules_found
    if not missing_mods:
        report.ok("safetensors", "Todos os target_modules presentes nos pesos", ", ".join(sorted(modules_found)))
    else:
        report.fail("safetensors", f"Modulos ausentes nos pesos", str(missing_mods))

    # Metadata format
    fmt = meta.get("__metadata__", {}).get("format", "")
    if fmt == "pt":
        report.ok("safetensors", "Formato dos pesos", "pt (PyTorch)")
    elif fmt:
        report.warn("safetensors", f"Formato inesperado: {fmt}")
    else:
        report.warn("safetensors", "Metadata de formato ausente")


def check_tokenizer(path: Path, report: Report):
    """4. Valida o tokenizer."""
    print("\n[4] Tokenizer")

    # tokenizer.json — verifica estrutura mínima
    tj_path = path / "tokenizer.json"
    if tj_path.exists():
        try:
            with open(tj_path, encoding="utf-8") as f:
                tj = json.load(f)
            vocab_size = len(tj.get("model", {}).get("vocab", {}))
            if vocab_size == 0:
                # Alguns tokenizers usam outro campo
                vocab_size = tj.get("added_tokens_decoder", {})
                vocab_size = len(vocab_size) if vocab_size else 0
            if vocab_size > 1000:
                report.ok("tokenizer", "tokenizer.json valido", f"vocabulario: {vocab_size:,} tokens")
            else:
                report.warn("tokenizer", "Vocabulario muito pequeno ou estrutura diferente", str(vocab_size))
        except Exception as e:
            report.fail("tokenizer", "Falha ao ler tokenizer.json", str(e))

    # tokenizer_config.json
    tc_path = path / "tokenizer_config.json"
    if tc_path.exists():
        try:
            with open(tc_path, encoding="utf-8") as f:
                tc = json.load(f)
            eos = tc.get("eos_token", "")
            bos = tc.get("bos_token", "")
            pad = tc.get("pad_token", "")
            if eos:
                report.ok("tokenizer", "eos_token definido", repr(eos))
            else:
                report.warn("tokenizer", "eos_token ausente no tokenizer_config")
            if bos:
                report.ok("tokenizer", "bos_token definido", repr(bos))
            if pad:
                report.ok("tokenizer", "pad_token definido", repr(pad))
            else:
                report.warn("tokenizer", "pad_token ausente — pode causar erros na inferencia")
        except Exception as e:
            report.fail("tokenizer", "Falha ao ler tokenizer_config.json", str(e))

    # tokenizer.model (opcional, sentencepiece)
    tm_path = path / "tokenizer.model"
    if tm_path.exists():
        size_kb = tm_path.stat().st_size / 1024
        if size_kb > 100:
            report.ok("tokenizer", "tokenizer.model (sentencepiece)", f"{size_kb:.0f} KB")
        else:
            report.warn("tokenizer", "tokenizer.model muito pequeno", f"{size_kb:.1f} KB")


def check_chat_template(path: Path, report: Report):
    """5. Valida o chat template."""
    print("\n[5] Chat Template")

    # Verifica no tokenizer_config.json
    tc_path = path / "tokenizer_config.json"
    has_template_in_config = False
    if tc_path.exists():
        try:
            with open(tc_path, encoding="utf-8") as f:
                tc = json.load(f)
            template = tc.get("chat_template", "")
            if template and len(template) > 20:
                report.ok("chat_template", "chat_template embutido no tokenizer_config.json", f"{len(template)} chars")
                has_template_in_config = True
        except Exception:
            pass

    # Verifica arquivo .jinja separado
    jinja_path = path / "chat_template.jinja"
    if jinja_path.exists():
        content = jinja_path.read_text(encoding="utf-8")
        if len(content) > 20:
            # Verifica tokens esperados do Phi-3
            has_user = "<|user|>" in content or "user" in content.lower()
            has_assistant = "<|assistant|>" in content or "assistant" in content.lower()
            if has_user and has_assistant:
                report.ok("chat_template", "chat_template.jinja valido", f"{len(content)} chars — tokens user/assistant presentes")
            else:
                report.warn("chat_template", "chat_template.jinja pode estar incompleto", "tokens user/assistant nao encontrados")
        else:
            report.warn("chat_template", "chat_template.jinja muito curto", f"{len(content)} chars")
    elif not has_template_in_config:
        report.warn("chat_template", "Nenhum chat_template encontrado", "respostas podem ficar sem formatacao correta")


def check_consistency(cfg: dict, path: Path, report: Report):
    """6. Consistência cruzada entre arquivos."""
    print("\n[6] Consistencia")

    if not cfg:
        report.warn("consistencia", "Sem adapter_config — verificacao de consistencia ignorada")
        return

    # adapter_config vs tensores — verifica se os target_modules batem
    st_path = path / "adapter_model.safetensors"
    if st_path.exists():
        try:
            with open(st_path, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                meta = json.loads(f.read(header_len).decode("utf-8"))
            tensors = [k for k in meta.keys() if k != "__metadata__"]
            target_in_config = set(cfg.get("target_modules", []))
            target_in_weights = set()
            for k in tensors:
                for mod in target_in_config:
                    if f".{mod}." in k:
                        target_in_weights.add(mod)

            if target_in_config == target_in_weights:
                report.ok("consistencia", "target_modules batem entre config e pesos")
            else:
                diff = target_in_config - target_in_weights
                report.warn("consistencia", f"Modulos no config sem pesos correspondentes", str(diff))
        except Exception as e:
            report.warn("consistencia", "Nao foi possivel verificar consistencia config/pesos", str(e))

    # inference_mode deve ser True
    if cfg.get("inference_mode") is True:
        report.ok("consistencia", "inference_mode=True — adaptadores prontos para inferencia")
    else:
        report.warn("consistencia", "inference_mode != True", "execute model.eval() antes de inferir")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    sep = "=" * 60
    ts  = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    print(f"\n{sep}")
    print("  VALIDACAO DOS ADAPTADORES LORA (Colab Fine-tuning)")
    print(f"  {ts}")
    print(sep)
    print(f"\nPasta: {MODEL_OUTPUT_DIR}\n")

    # Verifica se a pasta existe
    if not MODEL_OUTPUT_DIR.exists():
        print(f"ERRO: Pasta nao encontrada: {MODEL_OUTPUT_DIR}")
        print("Copie os arquivos gerados no Colab para esta pasta e execute novamente.")
        sys.exit(1)

    report = Report()

    check_files(MODEL_OUTPUT_DIR, report)
    cfg = check_adapter_config(MODEL_OUTPUT_DIR, report)
    check_safetensors(MODEL_OUTPUT_DIR, report)
    check_tokenizer(MODEL_OUTPUT_DIR, report)
    check_chat_template(MODEL_OUTPUT_DIR, report)
    check_consistency(cfg, MODEL_OUTPUT_DIR, report)

    # ── Relatório Final ──────────────────────────────────────
    s = report.summary()
    print(f"\n{sep}")
    print("  RESULTADO DA VALIDACAO")
    print(sep)
    print(f"  Total de verificacoes: {s['total']}")
    print(f"  Aprovadas:  {s['ok']}")
    print(f"  Avisos:     {s['warnings']}")
    print(f"  Falhas:     {s['failures']}")
    print()

    if report.approved():
        print("  APROVADO — Adaptadores validos para uso no projeto.")
        if s["warnings"] > 0:
            print(f"  ({s['warnings']} aviso(s) nao criticos — verifique acima)")
    else:
        print("  REPROVADO — Corrija as falhas antes de prosseguir.")
        print()
        print("  Falhas encontradas:")
        for c in report.checks:
            if c["status"] == "FAIL":
                detail = f" -> {c['detail']}" if c["detail"] else ""
                print(f"    x {c['msg']}{detail}")

    print(f"\n{sep}")

    # Salva relatório JSON
    out_path = MODEL_OUTPUT_DIR.parent / "validation_report.json"
    report_data = {
        "timestamp": ts,
        "model_path": str(MODEL_OUTPUT_DIR),
        "approved": report.approved(),
        "summary": s,
        "checks": report.checks,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"  Relatorio salvo em: {out_path}")
    print(sep)

    sys.exit(0 if report.approved() else 1)


if __name__ == "__main__":
    main()
