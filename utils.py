"""
utils.py — Utilitários compartilhados do MedAssist.
"""
import sys
import io

_encoding_fixed = False


def fix_windows_encoding():
    """
    Força UTF-8 no stdout/stderr do Windows (resolve UnicodeEncodeError com emojis).
    Seguro para ser chamado múltiplas vezes — executa apenas na primeira chamada.
    """
    global _encoding_fixed
    if _encoding_fixed:
        return
    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, "buffer"):
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer, encoding="utf-8", errors="replace"
                )
            if hasattr(sys.stderr, "buffer"):
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer, encoding="utf-8", errors="replace"
                )
        except Exception:
            pass  # Se falhar, segue sem o fix (melhor do que travar)
    _encoding_fixed = True
