"""Subrotina de versão."""

from __future__ import annotations

import subprocess
import sys

NOME = "difusao"
DESCRICAO = "Solver de difusão de nêutrons monoenergética 1D/2D (Opções 1 e 3)"
VERSAO = "1.0.0"
DATA_VERSAO = "2026-06-09"
AUTOR = "Projeto final — Física de Reatores"


def _git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "n/d"


def imprime_versao(arq=sys.stdout) -> None:
    largura = 64
    linhas = [
        "=" * largura,
        f"{NOME} v{VERSAO}  ({DATA_VERSAO})",
        DESCRICAO,
        f"commit git: {_git_hash()}    python: {sys.version.split()[0]}",
        AUTOR,
        "=" * largura,
    ]
    print("\n".join(linhas), file=arq)
