"""Estilo gráfico (Seaborn + fontes LaTeX, com fallback mathtext)."""

from __future__ import annotations

import shutil

import matplotlib.pyplot as plt
import seaborn as sns

PALETA = "crest"


def latex_disponivel() -> bool:
    return shutil.which("latex") is not None and shutil.which("dvipng") is not None


def aplicar_estilo(usetex: bool | None = None, contexto: str = "notebook") -> bool:
    """Configura Seaborn + fontes; retorna o modo usetex efetivo."""
    sns.set_theme(style="ticks", context=contexto, palette=PALETA)

    if usetex is None:
        usetex = latex_disponivel()

    comum = {
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
    if usetex:
        plt.rcParams.update({
            **comum,
            "text.usetex": True,
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}",
        })
    else:
        plt.rcParams.update({
            **comum,
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "font.serif": ["DejaVu Serif"],
            "axes.formatter.use_mathtext": True,
        })
    return usetex
