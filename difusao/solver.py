"""Solvers de fonte fixa (LU/CG) e de autovalor (power / Wielandt 2 estágios)."""

from __future__ import annotations

import dataclasses

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_PERMC = "MMD_AT_PLUS_A"


@dataclasses.dataclass
class ResultadoFonteFixa:
    phi: np.ndarray
    metodo: str
    iteracoes: int


@dataclasses.dataclass
class ResultadoAutovalor:
    phi: np.ndarray
    k_eff: float
    metodo: str
    iteracoes: int
    fatoracoes: int
    historico: list


def resolver_fonte_fixa(A, q, metodo: str = "direto",
                        tol: float = 1.0e-10) -> ResultadoFonteFixa:
    """Resolve A phi = q por LU esparsa ('direto') ou CG+Jacobi ('cg')."""
    if metodo == "direto":
        lu = spla.splu(A, permc_spec=_PERMC)
        return ResultadoFonteFixa(phi=lu.solve(q), metodo="direto (LU/MMD)",
                                  iteracoes=0)

    M = sp.diags(1.0 / A.diagonal())
    contador = [0]
    phi, info = spla.cg(A, q, rtol=tol, atol=0.0, M=M, maxiter=50000,
                        callback=lambda _: contador.__setitem__(0, contador[0] + 1))
    if info != 0:
        lu = spla.splu(A, permc_spec=_PERMC)
        return ResultadoFonteFixa(phi=lu.solve(q),
                                  metodo="direto (fallback após CG não convergir)",
                                  iteracoes=contador[0])
    return ResultadoFonteFixa(phi=phi, metodo="cg + Jacobi",
                              iteracoes=contador[0])


def _erro_forma(phi_novo: np.ndarray, phi_velho: np.ndarray) -> float:
    a = phi_novo / np.linalg.norm(phi_novo)
    b = phi_velho / np.linalg.norm(phi_velho)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(a)))


def _power_basico(lu, F, phi, k, n_iters, historico, base_iter,
                  tol_k=None, tol_phi=None):
    f = F @ phi
    for it in range(1, n_iters + 1):
        phi_novo = lu.solve(f / k)
        f_novo = F @ phi_novo
        k_novo = k * f_novo.sum() / f.sum()
        err_k = abs(k_novo - k) / abs(k_novo)
        err_phi = _erro_forma(phi_novo, phi)
        historico.append((base_iter + it, k_novo, err_k, err_phi))
        phi, k, f = phi_novo, k_novo, f_novo
        if tol_k is not None and err_k < tol_k and err_phi < tol_phi and it > 3:
            return phi, k, base_iter + it, True
    return phi, k, base_iter + n_iters, False


def resolver_autovalor(A, F, metodo: str = "wielandt",
                       tol_k: float = 1.0e-8, tol_phi: float = 1.0e-6,
                       max_iter: int = 50000,
                       semente: int = 42) -> ResultadoAutovalor:
    """Resolve A phi = (1/k) F phi para o modo fundamental ('power' ou 'wielandt')."""
    n = A.shape[0]
    rng = np.random.default_rng(semente)
    phi = 1.0 + 0.5 * rng.random(n)
    k = 1.0
    historico: list = []

    lu = spla.splu(A, permc_spec=_PERMC)
    fatoracoes = 1

    if metodo == "power":
        phi, k, total, ok = _power_basico(lu, F, phi, k, max_iter, historico, 0,
                                          tol_k=tol_k, tol_phi=tol_phi)
        if not ok:
            raise RuntimeError(f"power iteration não convergiu em {max_iter} iterações.")
        return ResultadoAutovalor(phi=phi, k_eff=k, metodo="power",
                                  iteracoes=total, fatoracoes=fatoracoes,
                                  historico=historico)

    n_livres = 8
    phi, k, total, _ = _power_basico(lu, F, phi, k, n_livres, historico, 0)

    for estagio, delta in enumerate((0.02, 0.002)):
        ultimo = estagio == 1
        ke = k + delta
        B = (A - F.multiply(1.0 / ke)).tocsc()
        luB = spla.splu(B, permc_spec=_PERMC)
        fatoracoes += 1

        k_shift = 1.0 / (1.0 / k - 1.0 / ke)
        f = F @ phi
        k_ant = k
        for _ in range(max_iter):
            phi_novo = luB.solve(f / k_shift)
            f_novo = F @ phi_novo
            k_shift = k_shift * f_novo.sum() / f.sum()
            k = 1.0 / (1.0 / k_shift + 1.0 / ke)
            err_k = abs(k - k_ant) / abs(k)
            err_phi = _erro_forma(phi_novo, phi)
            total += 1
            historico.append((total, k, err_k, err_phi))
            phi, f, k_ant = phi_novo, f_novo, k
            if err_k < tol_k and err_phi < tol_phi:
                if ultimo:
                    return ResultadoAutovalor(
                        phi=phi, k_eff=k, metodo="wielandt (2 estágios)",
                        iteracoes=total, fatoracoes=fatoracoes,
                        historico=historico)
                break
            if not ultimo and err_k < 1.0e-4:
                break

    raise RuntimeError(f"Wielandt não convergiu em {max_iter} iterações por estágio.")
