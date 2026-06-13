"""Balanço de nêutrons (fonte = absorção + fuga), verificado a cada execução."""

from __future__ import annotations

import dataclasses

import numpy as np

_VACUO = "vacuo"


@dataclasses.dataclass
class Balanco:
    fonte: float
    absorcao: float
    fuga: float

    @property
    def residuo_relativo(self) -> float:
        ref = self.fonte if self.fonte > 0 else max(self.absorcao + self.fuga, 1e-300)
        return abs(self.absorcao + self.fuga - self.fonte) / ref


def _phi_borda(phi_celula, D_celula, h_celula):
    return phi_celula * 4.0 * D_celula / (h_celula + 4.0 * D_celula)


def balanco_1d(faces, mat, phi, bc_esq, bc_dir,
               k_eff: float | None = None) -> Balanco:
    """Balanço em 1D; em modo autovalor a 'fonte' é a fissão / k_eff."""
    dx = np.diff(faces)
    sigma_a = mat["sigma_t"] - mat["sigma_s"]
    D = 1.0 / (3.0 * mat["sigma_t"])

    absorcao = float((sigma_a * phi * dx).sum())
    if k_eff is None:
        fonte = float((mat["fonte"] * dx).sum())
    else:
        fonte = float((mat["nu_sigma_f"] * phi * dx).sum() / k_eff)

    fuga = 0.0
    if bc_esq == _VACUO:
        fuga += 0.5 * _phi_borda(phi[0], D[0], dx[0])
    if bc_dir == _VACUO:
        fuga += 0.5 * _phi_borda(phi[-1], D[-1], dx[-1])
    return Balanco(fonte=fonte, absorcao=absorcao, fuga=float(fuga))


def balanco_2d(faces_x, faces_y, mat, phi, bc_esq, bc_dir, bc_inf, bc_sup,
               k_eff: float | None = None) -> Balanco:
    """Balanço em 2D; phi tem forma (Ny, Nx)."""
    dx = np.diff(faces_x)
    dy = np.diff(faces_y)
    areas = dy[:, None] * dx[None, :]
    sigma_a = mat["sigma_t"] - mat["sigma_s"]
    D = 1.0 / (3.0 * mat["sigma_t"])

    absorcao = float((sigma_a * phi * areas).sum())
    if k_eff is None:
        fonte = float((mat["fonte"] * areas).sum())
    else:
        fonte = float((mat["nu_sigma_f"] * phi * areas).sum() / k_eff)

    fuga = 0.0
    if bc_esq == _VACUO:
        fuga += 0.5 * (_phi_borda(phi[:, 0], D[:, 0], dx[0]) * dy).sum()
    if bc_dir == _VACUO:
        fuga += 0.5 * (_phi_borda(phi[:, -1], D[:, -1], dx[-1]) * dy).sum()
    if bc_inf == _VACUO:
        fuga += 0.5 * (_phi_borda(phi[0, :], D[0, :], dy[0]) * dx).sum()
    if bc_sup == _VACUO:
        fuga += 0.5 * (_phi_borda(phi[-1, :], D[-1, :], dy[-1]) * dx).sum()
    return Balanco(fonte=fonte, absorcao=absorcao, fuga=float(fuga))
