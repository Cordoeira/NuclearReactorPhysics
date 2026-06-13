"""Montagem dos sistemas de difusão por volumes finitos (1D e 2D, CSC, SPD)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

_VACUO = "vacuo"


def montar_1d(faces: np.ndarray, mat: dict[str, np.ndarray],
              bc_esq: str, bc_dir: str):
    """Monta A (CSC), vetor de fonte q e operador de fissão F em 1D."""
    sigma_t, sigma_s = mat["sigma_t"], mat["sigma_s"]
    dx = np.diff(faces)
    sigma_a = sigma_t - sigma_s
    D = 1.0 / (3.0 * sigma_t)

    dx_e, dx_d = dx[:-1], dx[1:]
    D_e, D_d = D[:-1], D[1:]
    D_face = (dx_e + dx_d) / (dx_e / D_e + dx_d / D_d)
    g = D_face / (0.5 * (dx_e + dx_d))

    diag = sigma_a * dx
    diag[:-1] += g
    diag[1:] += g
    if bc_esq == _VACUO:
        diag[0] += 2.0 * D[0] / (dx[0] + 4.0 * D[0])
    if bc_dir == _VACUO:
        diag[-1] += 2.0 * D[-1] / (dx[-1] + 4.0 * D[-1])

    A = sp.diags([-g, diag, -g], [-1, 0, 1], format="csc")
    q = mat["fonte"] * dx
    F = sp.diags([mat["nu_sigma_f"] * dx], [0], format="csc")
    return A, q, F


def montar_2d(faces_x: np.ndarray, faces_y: np.ndarray,
              mat: dict[str, np.ndarray],
              bc_esq: str, bc_dir: str, bc_inf: str, bc_sup: str):
    """Monta A (CSC), q e F em 2D com indexação k = i + Nx*j; mat tem forma (Ny, Nx)."""
    sigma_t, sigma_s = mat["sigma_t"], mat["sigma_s"]
    Nx = len(faces_x) - 1
    Ny = len(faces_y) - 1
    dx = np.diff(faces_x)
    dy = np.diff(faces_y)
    sigma_a = sigma_t - sigma_s
    D = 1.0 / (3.0 * sigma_t)

    dx_e, dx_d = dx[:-1], dx[1:]
    D_e, D_d = D[:, :-1], D[:, 1:]
    Dfx = (dx_e + dx_d) / (dx_e / D_e + dx_d / D_d)
    g_x = Dfx * dy[:, None] / (0.5 * (dx_e + dx_d))

    dy_i, dy_s = dy[:-1], dy[1:]
    D_i, D_s = D[:-1, :], D[1:, :]
    Dfy = (dy_i[:, None] + dy_s[:, None]) / (dy_i[:, None] / D_i + dy_s[:, None] / D_s)
    g_y = Dfy * dx[None, :] / (0.5 * (dy_i + dy_s))[:, None]

    diag = (sigma_a * dy[:, None] * dx[None, :]).copy()
    diag[:, :-1] += g_x
    diag[:, 1:] += g_x
    diag[:-1, :] += g_y
    diag[1:, :] += g_y
    if bc_esq == _VACUO:
        diag[:, 0] += 2.0 * D[:, 0] * dy / (dx[0] + 4.0 * D[:, 0])
    if bc_dir == _VACUO:
        diag[:, -1] += 2.0 * D[:, -1] * dy / (dx[-1] + 4.0 * D[:, -1])
    if bc_inf == _VACUO:
        diag[0, :] += 2.0 * D[0, :] * dx / (dy[0] + 4.0 * D[0, :])
    if bc_sup == _VACUO:
        diag[-1, :] += 2.0 * D[-1, :] * dx / (dy[-1] + 4.0 * D[-1, :])

    N = Nx * Ny
    k = np.arange(N).reshape(Ny, Nx)

    rows = [k.ravel()]
    cols = [k.ravel()]
    vals = [diag.ravel()]

    ko, ke = k[:, :-1].ravel(), k[:, 1:].ravel()
    vx = -g_x.ravel()
    rows += [ko, ke]
    cols += [ke, ko]
    vals += [vx, vx]

    ks, kn = k[:-1, :].ravel(), k[1:, :].ravel()
    vy = -g_y.ravel()
    rows += [ks, kn]
    cols += [kn, ks]
    vals += [vy, vy]

    A = sp.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N),
    )
    areas = dy[:, None] * dx[None, :]
    q = (mat["fonte"] * areas).ravel()
    F = sp.diags([(mat["nu_sigma_f"] * areas).ravel()], [0], format="csc")
    return A, q, F
