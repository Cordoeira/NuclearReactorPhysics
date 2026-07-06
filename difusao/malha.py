"""Geração de malha cell-centered e mapeamento de materiais por célula."""

from __future__ import annotations

import sys

import numpy as np

from .input_data import ErroDeInput


def gerar_faces(L: float, N: int, graduacao: float = 1.0) -> np.ndarray:
    """N+1 faces em [0, L]; graduacao=p>1 concentra células perto de x=L."""
    s = np.linspace(0.0, 1.0, N + 1)
    if graduacao == 1.0:
        return L * s
    return L * (1.0 - (1.0 - s) ** graduacao)


def centros(faces: np.ndarray) -> np.ndarray:
    return 0.5 * (faces[:-1] + faces[1:])


def _propriedade_por_celula_1d(regioes: list[dict], x_c: np.ndarray,
                               chave: str) -> np.ndarray:
    valores = np.full(x_c.shape, np.nan)
    for reg in regioes:
        x0, x1 = reg["x"]
        dentro = (x_c >= x0) & (x_c <= x1)
        valores[dentro] = reg[chave]
    if np.isnan(valores).any():
        faltam = x_c[np.isnan(valores)]
        raise ErroDeInput(
            "células sem material: as regiões não cobrem todo o domínio "
            f"(primeiro centróide descoberto: x = {faltam[0]:.6g})."
        )
    return valores


def _avisar_interface_no_centroide(regioes: list[dict], coords: np.ndarray,
                                   eixo: str) -> None:
    """Avisa quando uma borda de região coincide com um centróide de célula."""
    for reg in regioes:
        if eixo not in reg:
            continue
        for borda in reg[eixo]:
            atol = 1e-12 * max(abs(borda), 1.0)
            if np.any(np.isclose(coords, borda, rtol=0.0, atol=atol)):
                print(f"[AVISO] borda da região '{reg.get('nome', '?')}' em "
                      f"{eixo} = {borda:g} coincide com um centróide de célula: "
                      "a célula segue a última região listada. Alinhe a "
                      "interface a uma face da malha para evitar ambiguidade.",
                      file=sys.stderr)


def materiais_1d(regioes: list[dict], x_c: np.ndarray) -> dict[str, np.ndarray]:
    _avisar_interface_no_centroide(regioes, x_c, "x")
    return {ch: _propriedade_por_celula_1d(regioes, x_c, ch)
            for ch in ("sigma_t", "sigma_s", "nu_sigma_f", "fonte")}


def _propriedade_por_celula_2d(regioes: list[dict], X: np.ndarray,
                               Y: np.ndarray, chave: str) -> np.ndarray:
    valores = np.full(X.shape, np.nan)
    for reg in regioes:
        x0, x1 = reg["x"]
        y0, y1 = reg["y"]
        dentro = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        valores[dentro] = reg[chave]
    if np.isnan(valores).any():
        j, i = np.argwhere(np.isnan(valores))[0]
        raise ErroDeInput(
            "células sem material: as regiões não cobrem todo o domínio "
            f"(primeiro centróide descoberto: x = {X[j, i]:.6g}, y = {Y[j, i]:.6g})."
        )
    return valores


def materiais_2d(regioes: list[dict], x_c: np.ndarray,
                 y_c: np.ndarray) -> dict[str, np.ndarray]:
    _avisar_interface_no_centroide(regioes, x_c, "x")
    _avisar_interface_no_centroide(regioes, y_c, "y")
    X, Y = np.meshgrid(x_c, y_c)
    return {ch: _propriedade_por_celula_2d(regioes, X, Y, ch)
            for ch in ("sigma_t", "sigma_s", "nu_sigma_f", "fonte")}
