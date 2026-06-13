"""Escrita dos resultados: resumo .txt (versão + eco + balanço) e fluxo .csv."""

from __future__ import annotations

import io
import pathlib

import numpy as np

from .balanco import Balanco
from .input_echo import eco_input
from .version import imprime_versao


def _resumo_fluxo(phi: np.ndarray) -> str:
    return (f"  phi_min = {phi.min():.8e}\n"
            f"  phi_max = {phi.max():.8e}\n"
            f"  phi_med = {phi.mean():.8e}")


def escrever_saida(dados: dict, x_c, y_c, phi: np.ndarray, bal: Balanco,
                   info_solver: str, iteracoes: int,
                   k_eff: float | None = None,
                   fatoracoes: int | None = None) -> tuple[str, str]:
    """Grava P_resumo.txt e P_fluxo.csv; retorna os dois caminhos."""
    prefixo = pathlib.Path(dados["saida"]["prefixo"])
    prefixo.parent.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    imprime_versao(buf)
    eco_input(dados, buf)
    print("RESULTADOS", file=buf)
    print("-" * 64, file=buf)
    print(f"  solver: {info_solver}", file=buf)
    if iteracoes:
        print(f"  iterações: {iteracoes}", file=buf)
    if fatoracoes is not None:
        print(f"  fatorações LU: {fatoracoes}", file=buf)
    if k_eff is not None:
        print(f"  k_eff = {k_eff:.8f}", file=buf)
    print(_resumo_fluxo(phi), file=buf)
    print("BALANÇO DE NÊUTRONS", file=buf)
    rotulo_fonte = "fissão/k" if k_eff is not None else "fonte"
    print(f"  {rotulo_fonte:<12} = {bal.fonte:.10e}", file=buf)
    print(f"  {'absorção':<12} = {bal.absorcao:.10e}", file=buf)
    print(f"  {'fuga':<12} = {bal.fuga:.10e}", file=buf)
    print(f"  resíduo relativo |A+F-S|/S = {bal.residuo_relativo:.3e}", file=buf)
    print("-" * 64, file=buf)

    arq_resumo = f"{prefixo}_resumo.txt"
    pathlib.Path(arq_resumo).write_text(buf.getvalue(), encoding="utf-8")

    arq_fluxo = f"{prefixo}_fluxo.csv"
    if phi.ndim == 1:
        tabela = np.column_stack([x_c, phi])
        np.savetxt(arq_fluxo, tabela, delimiter=",", header="x,phi", comments="")
    else:
        X, Y = np.meshgrid(x_c, y_c)
        tabela = np.column_stack([X.ravel(), Y.ravel(), phi.ravel()])
        np.savetxt(arq_fluxo, tabela, delimiter=",", header="x,y,phi", comments="")

    return arq_resumo, arq_fluxo
