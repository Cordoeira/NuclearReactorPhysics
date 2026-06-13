"""Subrotina de eco do input validado (também gravada no resumo de saída)."""

from __future__ import annotations

import sys


def _linha(rotulo: str, valor, arq) -> None:
    print(f"  {rotulo:<22}: {valor}", file=arq)


def eco_input(dados: dict, arq=sys.stdout) -> None:
    dim = dados["dimensao"]
    print("-" * 64, file=arq)
    print("ECO DO INPUT", file=arq)
    print("-" * 64, file=arq)
    _linha("arquivo", dados["_arquivo"], arq)
    _linha("título", dados["titulo"], arq)
    _linha("dimensão", f"{dim}D", arq)
    _linha("modo", dados["modo"], arq)

    malha = dados["malha"]
    print("  malha:", file=arq)
    _linha("  Lx / Nx", f'{malha["Lx"]} cm / {malha["Nx"]} células', arq)
    if malha["graduacao_x"] != 1.0:
        _linha("  graduação em x", f'p = {malha["graduacao_x"]}', arq)
    if dim == 2:
        _linha("  Ly / Ny", f'{malha["Ly"]} cm / {malha["Ny"]} células', arq)
        if malha["graduacao_y"] != 1.0:
            _linha("  graduação em y", f'p = {malha["graduacao_y"]}', arq)
        _linha("  incógnitas", malha["Nx"] * malha["Ny"], arq)

    bc = dados["bc"]
    lados = ["esquerda", "direita"] + (["inferior", "superior"] if dim == 2 else [])
    print("  condições de contorno:", file=arq)
    for lado in lados:
        _linha(f"  {lado}", bc[lado], arq)

    print(f"  regiões ({len(dados['regioes'])}):", file=arq)
    for reg in dados["regioes"]:
        sigma_a = reg["sigma_t"] - reg["sigma_s"]
        c = reg["sigma_s"] / reg["sigma_t"]
        difusao = 1.0 / (3.0 * reg["sigma_t"])
        faixa = f'x∈{reg["x"]}' + (f', y∈{reg["y"]}' if dim == 2 else "")
        print(f"    - {reg['nome']:<12} {faixa}", file=arq)
        print(f"        sigma_t = {reg['sigma_t']:<8g} sigma_s = {reg['sigma_s']:<8g} "
              f"-> sigma_a = {sigma_a:.6g}, c = {c:.6g}, D = {difusao:.6g}", file=arq)
        print(f"        nu_sigma_f = {reg['nu_sigma_f']:<8g} fonte = {reg['fonte']:g}",
              file=arq)

    slv = dados["solver"]
    print("  solver:", file=arq)
    _linha("  método linear", slv["metodo"], arq)
    if slv["metodo"] == "cg":
        _linha("  tolerância cg", slv["tol"], arq)
    if dados["modo"] == "autovalor":
        av = slv["autovalor"]
        _linha("  autovalor", av["metodo"], arq)
        _linha("  tol_k / tol_phi", f'{av["tol_k"]} / {av["tol_phi"]}', arq)

    _linha("prefixo de saída", dados["saida"]["prefixo"], arq)
    print("-" * 64, file=arq)
