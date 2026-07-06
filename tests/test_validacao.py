"""Suíte de validação do solver de difusão (pytest)."""

import numpy as np
import pytest

from difusao.balanco import balanco_1d, balanco_2d
from difusao.input_data import ErroDeInput, ler_input
from difusao.malha import centros, gerar_faces, materiais_1d, materiais_2d
from difusao.montagem import montar_1d, montar_2d
from difusao.solver import resolver_autovalor, resolver_fonte_fixa


def _mat_uniforme(N, sigma_t, sigma_s, nu_sf=0.0, fonte=1.0):
    return {
        "sigma_t": np.full(N, sigma_t),
        "sigma_s": np.full(N, sigma_s),
        "nu_sigma_f": np.full(N, nu_sf),
        "fonte": np.full(N, fonte),
    }


def _analitico_meiaplaca(x, L, sigma_t, sigma_s, Q):
    """Meia-placa: reflexiva em x=0, vácuo (Marshak) em x=L, fonte uniforme."""
    D = 1.0 / (3.0 * sigma_t)
    sigma_a = sigma_t - sigma_s
    Ld = np.sqrt(D / sigma_a)
    den = np.cosh(L / Ld) + (2.0 * D / Ld) * np.sinh(L / Ld)
    return (Q / sigma_a) * (1.0 - np.cosh(x / Ld) / den)


def test_meio_infinito():
    """Reflexão dos dois lados emula meio infinito: phi = Q/Sigma_a."""
    N = 200
    faces = gerar_faces(10.0, N)
    mat = _mat_uniforme(N, sigma_t=1.0, sigma_s=0.5, fonte=1.0)
    A, q, _ = montar_1d(faces, mat, "reflexiva", "reflexiva")
    phi = resolver_fonte_fixa(A, q).phi
    assert np.allclose(phi, 2.0, rtol=1e-11)


def test_convergencia_ordem_2():
    """Erro contra a solução analítica deve cair como O(h^2)."""
    L, St, Ss, Q = 5.0, 1.0, 0.5, 1.0
    erros = []
    Ns = [40, 80, 160, 320]
    for N in Ns:
        faces = gerar_faces(L, N)
        x_c = centros(faces)
        mat = _mat_uniforme(N, St, Ss, fonte=Q)
        A, q, _ = montar_1d(faces, mat, "reflexiva", "vacuo")
        phi = resolver_fonte_fixa(A, q).phi
        erros.append(np.max(np.abs(phi - _analitico_meiaplaca(x_c, L, St, Ss, Q))))
    ordens = np.log2(np.array(erros[:-1]) / np.array(erros[1:]))
    assert np.all(ordens > 1.85), f"ordens medidas: {ordens}"


@pytest.mark.parametrize("s1", [0.8, 0.9, 0.99, 0.999, 0.9999])
@pytest.mark.parametrize("s2", [0.8, 0.99])
def test_balanco_10_casos(s1, s2):
    """Os 10 problemas obrigatórios conservam nêutrons em ~1e-13."""
    N, L = 400, 20.0
    faces = gerar_faces(L, N)
    x_c = centros(faces)
    metade = x_c < L / 2
    mat = {
        "sigma_t": np.ones(N),
        "sigma_s": np.where(metade, s1, s2),
        "nu_sigma_f": np.zeros(N),
        "fonte": np.where(metade, 1.0, 0.0),
    }
    A, q, _ = montar_1d(faces, mat, "vacuo", "vacuo")
    phi = resolver_fonte_fixa(A, q).phi
    bal = balanco_1d(faces, mat, phi, "vacuo", "vacuo")
    assert bal.residuo_relativo < 1e-11
    assert abs(bal.fonte - 10.0) < 1e-12


def test_validacao_cruzada_1d_2d():
    """2D com reflexão em y reproduz o perfil 1D (mesma física)."""
    Lx, Nx, Ny = 20.0, 200, 4
    faces_x = gerar_faces(Lx, Nx)
    faces_y = gerar_faces(1.0, Ny)
    x_c = centros(faces_x)
    metade = x_c < Lx / 2

    mat1 = {
        "sigma_t": np.ones(Nx),
        "sigma_s": np.where(metade, 0.99, 0.8),
        "nu_sigma_f": np.zeros(Nx),
        "fonte": np.where(metade, 1.0, 0.0),
    }
    A1, q1, _ = montar_1d(faces_x, mat1, "vacuo", "vacuo")
    phi1 = resolver_fonte_fixa(A1, q1).phi

    mat2 = {ch: np.tile(v, (Ny, 1)) for ch, v in mat1.items()}
    A2, q2, _ = montar_2d(faces_x, faces_y, mat2,
                          "vacuo", "vacuo", "reflexiva", "reflexiva")
    phi2 = resolver_fonte_fixa(A2, q2).phi.reshape(Ny, Nx)

    assert np.max(np.abs(phi2.mean(axis=0) - phi1)) < 1e-9
    bal = balanco_2d(faces_x, faces_y, mat2, phi2,
                     "vacuo", "vacuo", "reflexiva", "reflexiva")
    assert bal.residuo_relativo < 1e-10


def _reator_dois_nucleos():
    Nr = 1200
    faces = gerar_faces(60.0, Nr)
    x_c = centros(faces)
    core = ((x_c > 5) & (x_c < 25)) | ((x_c > 35) & (x_c < 55))
    mat = {
        "sigma_t": np.ones(Nr),
        "sigma_s": np.where(core, 0.70, 0.95),
        "nu_sigma_f": np.where(core, 0.39, 0.0),
        "fonte": np.zeros(Nr),
    }
    return montar_1d(faces, mat, "vacuo", "vacuo")


def test_autovalor_wielandt_igual_power():
    """Wielandt e power iteration dão o mesmo k; Wielandt itera menos."""
    A, _, F = _reator_dois_nucleos()
    res_w = resolver_autovalor(A, F, metodo="wielandt")
    res_p = resolver_autovalor(A, F, metodo="power")
    assert abs(res_w.k_eff - res_p.k_eff) < 5e-7
    assert res_w.iteracoes < res_p.iteracoes / 3
    assert res_w.iteracoes < 60
    assert 1.27 < res_w.k_eff < 1.29


def test_simetria_quarto_de_dominio():
    """Reflexão = espelho: 1/4 de domínio == domínio cheio espelhado."""
    Lq, Nq = 20.0, 40
    faces_q = gerar_faces(Lq, Nq)
    xq = centros(faces_q)
    Xq, Yq = np.meshgrid(xq, xq)
    dentro = (Xq < 10) & (Yq < 10)
    mat_q = {
        "sigma_t": np.ones_like(Xq),
        "sigma_s": np.where(dentro, 0.99, 0.8),
        "nu_sigma_f": np.zeros_like(Xq),
        "fonte": np.where(dentro, 1.0, 0.0),
    }
    Aq, qq, _ = montar_2d(faces_q, faces_q, mat_q,
                          "vacuo", "reflexiva", "vacuo", "reflexiva")
    phi_q = resolver_fonte_fixa(Aq, qq).phi.reshape(Nq, Nq)

    Nf = 2 * Nq
    faces_f = gerar_faces(2 * Lq, Nf)
    xf = centros(faces_f)
    Xf, Yf = np.meshgrid(xf, xf)
    esp = lambda u: np.minimum(u, 2 * Lq - u)
    dentro_f = (esp(Xf) < 10) & (esp(Yf) < 10)
    mat_f = {
        "sigma_t": np.ones_like(Xf),
        "sigma_s": np.where(dentro_f, 0.99, 0.8),
        "nu_sigma_f": np.zeros_like(Xf),
        "fonte": np.where(dentro_f, 1.0, 0.0),
    }
    Af, qf, _ = montar_2d(faces_f, faces_f, mat_f,
                          "vacuo", "vacuo", "vacuo", "vacuo")
    phi_f = resolver_fonte_fixa(Af, qf).phi.reshape(Nf, Nf)

    assert np.max(np.abs(phi_f[:Nq, :Nq] - phi_q)) < 1e-10


def test_input_invalido_detectado(tmp_path):
    """A validação de input rejeita sigma_s > sigma_t com mensagem clara."""
    ruim = tmp_path / "ruim.yaml"
    ruim.write_text(
        "dimensao: 1\n"
        "malha: {Lx: 10.0, Nx: 50}\n"
        "bc: {esquerda: vacuo, direita: vacuo}\n"
        "regioes:\n"
        "  - {x: [0.0, 10.0], sigma_t: 1.0, sigma_s: 1.5, fonte: 1.0}\n",
        encoding="utf-8")
    with pytest.raises(ErroDeInput, match="sigma_s"):
        ler_input(ruim)


def test_input_regioes_nao_cobrem(tmp_path):
    """Domínio com buraco de material é detectado na geração da malha."""
    inp = tmp_path / "buraco.yaml"
    inp.write_text(
        "dimensao: 1\n"
        "malha: {Lx: 10.0, Nx: 50}\n"
        "bc: {esquerda: vacuo, direita: vacuo}\n"
        "regioes:\n"
        "  - {x: [0.0, 4.0], sigma_t: 1.0, sigma_s: 0.5, fonte: 1.0}\n",
        encoding="utf-8")
    dados = ler_input(inp)
    faces = gerar_faces(10.0, 50)
    with pytest.raises(ValueError, match="não cobrem"):
        materiais_1d(dados["regioes"], centros(faces))


def test_cg_jacobi_concorda_com_lu():
    """O solver CG+Jacobi (solver.metodo: cg) reproduz a solução direta."""
    Nq = 80
    faces = gerar_faces(20.0, Nq)
    xc = centros(faces)
    X, Y = np.meshgrid(xc, xc)
    dentro = (X < 10) & (Y < 10)
    mat = {
        "sigma_t": np.ones_like(X),
        "sigma_s": np.where(dentro, 0.99, 0.8),
        "nu_sigma_f": np.zeros_like(X),
        "fonte": np.where(dentro, 1.0, 0.0),
    }
    A, q, _ = montar_2d(faces, faces, mat,
                        "vacuo", "reflexiva", "vacuo", "reflexiva")
    r_lu = resolver_fonte_fixa(A, q, metodo="direto")
    r_cg = resolver_fonte_fixa(A, q, metodo="cg", tol=1e-10)
    assert "cg" in r_cg.metodo
    assert r_cg.iteracoes > 0
    assert np.max(np.abs(r_lu.phi - r_cg.phi)) < 1e-8
