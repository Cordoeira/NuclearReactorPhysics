"""Subrotina de leitura e validação dos dados de entrada (YAML/JSON)."""

from __future__ import annotations

import copy
import json
import pathlib
import sys
from typing import Any

import yaml

_BCS_VALIDAS = {"vacuo", "reflexiva"}
_DEFAULTS: dict[str, Any] = {
    "titulo": "(sem título)",
    "modo": "fonte_fixa",
    "solver": {
        "metodo": "direto",
        "tol": 1.0e-10,
        "autovalor": {"metodo": "wielandt", "tol_k": 1.0e-8, "tol_phi": 1.0e-6},
    },
}


class ErroDeInput(ValueError):
    """Erro de validação do arquivo de entrada."""


def _mesclar_defaults(dados: dict, defaults: dict) -> dict:
    saida = copy.deepcopy(defaults)
    for chave, valor in dados.items():
        if isinstance(valor, dict) and isinstance(saida.get(chave), dict):
            saida[chave] = _mesclar_defaults(valor, saida[chave])
        else:
            saida[chave] = copy.deepcopy(valor)
    return saida


def _exigir(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise ErroDeInput(mensagem)


def _numero(valor, campo: str) -> float:
    """Converte para float exigindo tipo numérico no input."""
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ErroDeInput(f"{campo}: deve ser numérico (recebido {valor!r}).")
    return float(valor)


def _inteiro(valor, campo: str) -> int:
    """Converte para int exigindo valor inteiro no input."""
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) \
            or int(valor) != valor:
        raise ErroDeInput(f"{campo}: deve ser inteiro (recebido {valor!r}).")
    return int(valor)


def _validar_regiao(reg: dict, indice: int, dimensao: int) -> dict:
    reg = dict(reg)
    reg.setdefault("nome", f"regiao{indice + 1}")
    reg.setdefault("nu_sigma_f", 0.0)
    reg.setdefault("fonte", 0.0)

    _exigir("x" in reg, f"regiões[{indice}]: faltou o intervalo 'x'.")
    _exigir(len(reg["x"]) == 2 and reg["x"][0] < reg["x"][1],
            f"regiões[{indice}]: 'x' deve ser [x_ini, x_fim] com x_ini < x_fim.")
    if dimensao == 2:
        _exigir("y" in reg, f"regiões[{indice}]: 2D exige o intervalo 'y'.")
        _exigir(len(reg["y"]) == 2 and reg["y"][0] < reg["y"][1],
                f"regiões[{indice}]: 'y' deve ser [y_ini, y_fim] com y_ini < y_fim.")

    _exigir("sigma_t" in reg and reg["sigma_t"] > 0,
            f"regiões[{indice}]: 'sigma_t' obrigatório e > 0.")
    _exigir("sigma_s" in reg and 0.0 <= reg["sigma_s"] <= reg["sigma_t"],
            f"regiões[{indice}]: exige 0 <= sigma_s <= sigma_t "
            f"(recebido sigma_s={reg.get('sigma_s')}, sigma_t={reg['sigma_t']}).")
    _exigir(reg["nu_sigma_f"] >= 0, f"regiões[{indice}]: nu_sigma_f >= 0.")
    _exigir(reg["fonte"] >= 0, f"regiões[{indice}]: fonte >= 0.")
    return reg


def ler_input(caminho: str | pathlib.Path) -> dict:
    """Lê, completa com defaults e valida o arquivo de entrada."""
    caminho = pathlib.Path(caminho)
    _exigir(caminho.exists(), f"arquivo de input não encontrado: {caminho}")

    texto = caminho.read_text(encoding="utf-8")
    try:
        if caminho.suffix.lower() == ".json":
            bruto = json.loads(texto)
        else:
            bruto = yaml.safe_load(texto)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ErroDeInput(f"falha ao interpretar {caminho}: {exc}") from None
    _exigir(isinstance(bruto, dict), "o input deve ser um mapeamento chave: valor.")

    dados = _mesclar_defaults(bruto, _DEFAULTS)
    dados["_arquivo"] = str(caminho)

    _exigir(dados.get("dimensao") in (1, 2), "campo 'dimensao' deve ser 1 ou 2.")
    dim = dados["dimensao"]
    _exigir(dados["modo"] in ("fonte_fixa", "autovalor"),
            "campo 'modo' deve ser 'fonte_fixa' ou 'autovalor'.")

    _exigir("malha" in dados, "faltou o bloco 'malha'.")
    malha = dados["malha"]
    malha["Lx"] = _numero(malha.get("Lx"), "malha.Lx")
    malha["Nx"] = _inteiro(malha.get("Nx"), "malha.Nx")
    _exigir(malha["Lx"] > 0 and malha["Nx"] >= 3, "malha: exige Lx > 0 e Nx >= 3.")
    malha.setdefault("graduacao_x", 1.0)
    _exigir(malha["graduacao_x"] >= 1.0, "malha: graduacao_x >= 1.")
    if dim == 2:
        malha["Ly"] = _numero(malha.get("Ly"), "malha.Ly")
        malha["Ny"] = _inteiro(malha.get("Ny"), "malha.Ny")
        _exigir(malha["Ly"] > 0 and malha["Ny"] >= 3,
                "malha 2D: exige Ly > 0 e Ny >= 3.")
        malha.setdefault("graduacao_y", 1.0)
        _exigir(malha["graduacao_y"] >= 1.0, "malha: graduacao_y >= 1.")

    _exigir("bc" in dados, "faltou o bloco 'bc'.")
    bc = dados["bc"]
    lados = ["esquerda", "direita"] + (["inferior", "superior"] if dim == 2 else [])
    for lado in lados:
        _exigir(bc.get(lado) in _BCS_VALIDAS,
                f"bc.{lado}: deve ser 'vacuo' ou 'reflexiva' (recebido {bc.get(lado)!r}).")

    _exigir(isinstance(dados.get("regioes"), list) and len(dados["regioes"]) >= 1,
            "faltou a lista 'regioes' (ao menos uma região).")
    dados["regioes"] = [_validar_regiao(r, i, dim)
                        for i, r in enumerate(dados["regioes"])]

    slv = dados["solver"]
    _exigir(slv["metodo"] in ("direto", "cg"),
            "solver.metodo deve ser 'direto' ou 'cg'.")
    _exigir(slv["autovalor"]["metodo"] in ("power", "wielandt"),
            "solver.autovalor.metodo deve ser 'power' ou 'wielandt'.")

    if dados["modo"] == "autovalor":
        _exigir(any(r["nu_sigma_f"] > 0 for r in dados["regioes"]),
                "modo 'autovalor' exige ao menos uma região com nu_sigma_f > 0.")
        if any(r["fonte"] > 0 for r in dados["regioes"]):
            print("[AVISO] modo 'autovalor': o campo 'fonte' das regiões é "
                  "ignorado (problema homogêneo de autovalor).", file=sys.stderr)

    dados.setdefault("saida", {})
    dados["saida"].setdefault("prefixo", f"outputs/{caminho.stem}")

    return dados
