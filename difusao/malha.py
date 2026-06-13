"""Geração de malha cell-centered e mapeamento de materiais por célula."""

from __future__ import annotations

import numpy as np


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
        raise ValueError(
            "células sem material: as regiões não cobrem todo o domínio "
            f"(primeiro centróide descoberto: x = {faltam[0]:.6g})."
        )
    return valores


def materiais_1d(regioes: list[dict], x_c: np.ndarray) -> dict[str, np.ndarray]:
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
        raise ValueError(
            "células sem material: as regiões não cobrem todo o domínio "
            f"(primeiro centróide descoberto: x = {X[j, i]:.6g}, y = {Y[j, i]:.6g})."
        )
    return valores


def materiais_2d(regioes: list[dict], x_c: np.ndarray,
                 y_c: np.ndarray) -> dict[str, np.ndarray]:
    X, Y = np.meshgrid(x_c, y_c)
    return {ch: _propriedade_por_celula_2d(regioes, X, Y, ch)
            for ch in ("sigma_t", "sigma_s", "nu_sigma_f", "fonte")}
