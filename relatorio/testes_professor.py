"""Bateria de avaliação 'do professor' — verificação, física, numérica e robustez.

Roda a partir da raiz do repositório:
    .venv/bin/python relatorio/testes_professor.py

Gera relatorio/dados/resultados.json com todos os números citados no relatório.
NÃO altera o código avaliado: apenas o exercita.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from difusao.balanco import balanco_1d, balanco_2d
from difusao.malha import centros, gerar_faces
from difusao.montagem import montar_1d, montar_2d
from difusao.solver import resolver_autovalor, resolver_fonte_fixa

RAIZ = pathlib.Path(__file__).resolve().parents[1]
PYTHON = str(RAIZ / ".venv" / "bin" / "python")
DADOS = RAIZ / "relatorio" / "dados"
DADOS.mkdir(parents=True, exist_ok=True)

R: dict = {}


def _mat(N, st, ss, nsf=0.0, q=0.0):
    return {"sigma_t": np.full(N, st), "sigma_s": np.full(N, ss),
            "nu_sigma_f": np.full(N, nsf), "fonte": np.full(N, q)}


# ============================================================ A. VERIFICAÇÃO
def a1_meio_infinito():
    N = 200
    faces = gerar_faces(10.0, N)
    mat = _mat(N, 1.0, 0.5, q=1.0)
    A, q, _ = montar_1d(faces, mat, "reflexiva", "reflexiva")
    phi = resolver_fonte_fixa(A, q).phi
    exato = 1.0 / 0.5
    R["A1_meio_infinito"] = {
        "phi_exato": exato,
        "erro_rel_max": float(np.max(np.abs(phi - exato)) / exato),
        "aprovado": bool(np.max(np.abs(phi - exato)) / exato < 1e-11),
    }


def _analitico_meiaplaca(x, L, st, ss, Q, d=None):
    """Meia-placa (reflexiva em 0, Robin com dist. extrapolada d em L)."""
    D = 1.0 / (3.0 * st)
    sa = st - ss
    Ld = np.sqrt(D / sa)
    if d is None:
        d = 2.0 * D                       # Marshak
    den = np.cosh(L / Ld) + (d / Ld) * np.sinh(L / Ld)
    return (Q / sa) * (1.0 - np.cosh(x / Ld) / den)


def a2_convergencia():
    L, st, ss, Q = 5.0, 1.0, 0.5, 1.0
    Ns = [40, 80, 160, 320, 640]
    erros = []
    for N in Ns:
        faces = gerar_faces(L, N)
        x_c = centros(faces)
        A, q, _ = montar_1d(faces, _mat(N, st, ss, q=Q), "reflexiva", "vacuo")
        phi = resolver_fonte_fixa(A, q).phi
        erros.append(float(np.max(np.abs(phi - _analitico_meiaplaca(x_c, L, st, ss, Q)))))
    ordens = [float(np.log2(erros[i] / erros[i + 1])) for i in range(len(Ns) - 1)]
    R["A2_convergencia"] = {"Ns": Ns, "erros": erros, "ordens": ordens,
                            "aprovado": bool(min(ordens) > 1.85)}


def _k_analitico_placa(L, st, ss, nsf):
    """Placa nua: tan(BL/2) = 1/(2DB)  ->  k = nsf/(sa + D B^2)."""
    D = 1.0 / (3.0 * st)
    sa = st - ss
    f = lambda B: np.tan(B * L / 2.0) - 1.0 / (2.0 * D * B)
    B = brentq(f, 1e-8, np.pi / L - 1e-8, xtol=1e-14)
    return nsf / (sa + D * B * B), B


def a3_placa_critica():
    L, st, ss, nsf = 20.0, 1.0, 0.9, 0.15
    k_ana, B = _k_analitico_placa(L, st, ss, nsf)
    Ns = [50, 100, 200, 400, 800]
    ks = []
    for N in Ns:
        faces = gerar_faces(L, N)
        A, _, F = montar_1d(faces, _mat(N, st, ss, nsf=nsf), "vacuo", "vacuo")
        ks.append(float(resolver_autovalor(A, F).k_eff))
    erros = [abs(k - k_ana) for k in ks]
    ordens = [float(np.log2(erros[i] / erros[i + 1])) for i in range(len(Ns) - 1)]
    # Richardson (ordem 2) com as duas malhas mais finas
    k_rich = ks[-1] + (ks[-1] - ks[-2]) / 3.0
    R["A3_placa_critica"] = {
        "L": L, "sigma_t": st, "sigma_s": ss, "nu_sigma_f": nsf,
        "B_analitico": float(B), "k_analitico": float(k_ana),
        "Ns": Ns, "k_eff": ks, "erros": erros, "ordens": ordens,
        "k_richardson": float(k_rich),
        "erro_richardson": float(abs(k_rich - k_ana)),
        "aprovado": bool(erros[-1] < 5e-7 and min(ordens) > 1.9),
    }


# ================================================================ B. FÍSICA
def b1_continuidade_corrente():
    """Reconstrói a corrente na interface de material pelos dois lados."""
    N, L = 400, 20.0
    faces = gerar_faces(L, N)
    x_c = centros(faces)
    metade = x_c < L / 2
    mat = {"sigma_t": np.ones(N), "sigma_s": np.where(metade, 0.99, 0.8),
           "nu_sigma_f": np.zeros(N), "fonte": np.where(metade, 1.0, 0.0)}
    A, q, _ = montar_1d(faces, mat, "vacuo", "vacuo")
    phi = resolver_fonte_fixa(A, q).phi
    dx = np.diff(faces)
    D = 1.0 / (3.0 * mat["sigma_t"])
    i = int(np.searchsorted(faces, L / 2)) - 1     # célula à esquerda da interface
    he, hd = dx[i] / 2.0, dx[i + 1] / 2.0
    ce, cd = D[i] / he, D[i + 1] / hd
    phi_f = (ce * phi[i] + cd * phi[i + 1]) / (ce + cd)
    J_esq = -D[i] * (phi_f - phi[i]) / he
    J_dir = -D[i + 1] * (phi[i + 1] - phi_f) / hd
    R["B1_corrente_interface"] = {
        "x_interface": float(faces[i + 1]),
        "J_esquerda": float(J_esq), "J_direita": float(J_dir),
        "salto_relativo": float(abs(J_esq - J_dir) / abs(J_esq)),
        "aprovado": bool(abs(J_esq - J_dir) / abs(J_esq) < 1e-12),
    }


def _reator_refletido(Nr=600, ss_refl=0.95):
    faces = gerar_faces(60.0, Nr)
    x_c = centros(faces)
    core = (x_c > 20) & (x_c < 40)
    mat = {"sigma_t": np.ones(Nr),
           "sigma_s": np.where(core, 0.70, ss_refl),
           "nu_sigma_f": np.where(core, 0.39, 0.0),
           "fonte": np.zeros(Nr)}
    return faces, mat


def b2_sensibilidade():
    faces, mat = _reator_refletido()
    A, _, F = montar_1d(faces, mat, "vacuo", "vacuo")
    k0 = resolver_autovalor(A, F).k_eff

    # (a) nu_sigma_f -> 1.01x : em 1 grupo, k escala EXATAMENTE 1.01x
    mat_p = dict(mat); mat_p["nu_sigma_f"] = 1.01 * mat["nu_sigma_f"]
    A1, _, F1 = montar_1d(faces, mat_p, "vacuo", "vacuo")
    k1 = resolver_autovalor(A1, F1).k_eff

    # (b) refletor mais absorvedor (sigma_s 0.95 -> 0.90) : k deve CAIR
    faces2, mat2 = _reator_refletido(ss_refl=0.90)
    A2, _, F2 = montar_1d(faces2, mat2, "vacuo", "vacuo")
    k2 = resolver_autovalor(A2, F2).k_eff

    R["B2_sensibilidade"] = {
        "k_base": float(k0), "k_nusf_1p01": float(k1),
        "razao_obtida": float(k1 / k0), "razao_esperada": 1.01,
        "desvio_escala": float(abs(k1 / k0 - 1.01)),
        "k_refletor_mais_absorvedor": float(k2),
        "k_caiu": bool(k2 < k0),
        "aprovado": bool(abs(k1 / k0 - 1.01) < 1e-6 and k2 < k0),
    }


def b3_efeito_refletor():
    # núcleo nu (20 cm)
    Nn = 200
    faces_n = gerar_faces(20.0, Nn)
    mat_n = _mat(Nn, 1.0, 0.70, nsf=0.39)
    An, _, Fn = montar_1d(faces_n, mat_n, "vacuo", "vacuo")
    rn = resolver_autovalor(An, Fn)
    core_n = np.ones(Nn, dtype=bool)

    # mesmo núcleo + 20 cm de refletor de cada lado
    faces_r, mat_r = _reator_refletido()
    Ar, _, Fr = montar_1d(faces_r, mat_r, "vacuo", "vacuo")
    rr = resolver_autovalor(Ar, Fr)
    x_r = centros(faces_r)
    core_r = (x_r > 20) & (x_r < 40)

    pico_n = float(rn.phi[core_n].max() / rn.phi[core_n].mean())
    pico_r = float(rr.phi[core_r].max() / rr.phi[core_r].mean())
    R["B3_efeito_refletor"] = {
        "k_nu": float(rn.k_eff), "k_refletido": float(rr.k_eff),
        "ganho_reatividade_pcm": float((rr.k_eff - rn.k_eff) * 1e5),
        "fator_pico_nu": pico_n, "fator_pico_refletido": pico_r,
        "aprovado": bool(rr.k_eff > rn.k_eff and pico_r < pico_n),
    }


def b4_dominancia():
    Nr = 1200
    faces = gerar_faces(60.0, Nr)
    x_c = centros(faces)
    core = ((x_c > 5) & (x_c < 25)) | ((x_c > 35) & (x_c < 55))
    mat = {"sigma_t": np.ones(Nr), "sigma_s": np.where(core, 0.70, 0.95),
           "nu_sigma_f": np.where(core, 0.39, 0.0), "fonte": np.zeros(Nr)}
    A, _, F = montar_1d(faces, mat, "vacuo", "vacuo")
    rw = resolver_autovalor(A, F, metodo="wielandt")
    rp = resolver_autovalor(A, F, metodo="power")
    R["B4_dominancia"] = {
        "k_wielandt": float(rw.k_eff), "k_power": float(rp.k_eff),
        "dif_k": float(abs(rw.k_eff - rp.k_eff)),
        "iter_wielandt": rw.iteracoes, "iter_power": rp.iteracoes,
        "fatoracoes_wielandt": rw.fatoracoes,
        "ganho": float(rp.iteracoes / rw.iteracoes),
        "aprovado": bool(abs(rw.k_eff - rp.k_eff) < 5e-7
                         and rw.iteracoes * 3 < rp.iteracoes),
    }


def b5_limite_difusivo():
    """Quantifica Marshak (d=2D) vs transporte (d=0.7104/Sigma_tr) no caso c->1."""
    L, st, Q = 10.0, 1.0, 1.0
    saida = {}
    for ss in (0.8, 0.9999):
        D = 1.0 / (3.0 * st)
        d_m = 2.0 * D
        d_t = 0.7104 * 3.0 * D            # 0.7104 * lambda_tr, lambda_tr = 3D
        x = np.linspace(0, L, 2001)
        pm = _analitico_meiaplaca(x, L, st, ss, Q, d=d_m)
        pt = _analitico_meiaplaca(x, L, st, ss, Q, d=d_t)
        saida[f"c_{ss}"] = {
            "d_marshak_cm": d_m, "d_transporte_cm": d_t,
            "dif_rel_max_phi": float(np.max(np.abs(pm - pt) / pt.max())),
            "dif_rel_phi_borda": float(abs(pm[-1] - pt[-1]) / pt[-1]),
        }
    R["B5_limite_difusivo"] = saida


# ============================================================== C. NUMÉRICA
def c1_pior_caso_combinado():
    """2D + malha graduada + autovalor: o balanço ainda fecha?"""
    Nx = Ny = 60
    faces_x = gerar_faces(40.0, Nx, graduacao=1.8)
    faces_y = gerar_faces(40.0, Ny, graduacao=1.4)
    x_c, y_c = centros(faces_x), centros(faces_y)
    X, Y = np.meshgrid(x_c, y_c)
    core = (X > 8) & (X < 32) & (Y > 8) & (Y < 32)
    mat = {"sigma_t": np.ones_like(X), "sigma_s": np.where(core, 0.70, 0.95),
           "nu_sigma_f": np.where(core, 0.39, 0.0), "fonte": np.zeros_like(X)}
    A, _, F = montar_2d(faces_x, faces_y, mat, "vacuo", "vacuo", "vacuo", "reflexiva")
    res = resolver_autovalor(A, F)
    phi = res.phi.reshape(Ny, Nx)
    bal = balanco_2d(faces_x, faces_y, mat, phi,
                     "vacuo", "vacuo", "vacuo", "reflexiva", k_eff=res.k_eff)
    R["C1_pior_caso"] = {
        "k_eff": float(res.k_eff), "residuo_balanco": float(bal.residuo_relativo),
        "iteracoes": res.iteracoes,
        "aprovado": bool(bal.residuo_relativo < 1e-8),
    }


def c2_independencia_malha():
    ks = {}
    for N in (150, 300, 600, 1200):
        faces, mat = _reator_refletido(Nr=N)
        A, _, F = montar_1d(faces, mat, "vacuo", "vacuo")
        ks[N] = float(resolver_autovalor(A, F).k_eff)
    Ns = sorted(ks)
    k1, k2, k3 = ks[Ns[-3]], ks[Ns[-2]], ks[Ns[-1]]
    ordem = float(np.log2(abs(k2 - k1) / abs(k3 - k2)))
    k_rich = k3 + (k3 - k2) / 3.0
    R["C2_independencia_malha"] = {
        "k_por_N": ks, "ordem_observada_k": ordem,
        "k_extrapolado_richardson": float(k_rich),
        "dif_k_malha_fina_vs_extrapolado": float(abs(k3 - k_rich)),
        "aprovado": bool(ordem > 1.8 and abs(k3 - k_rich) < 1e-6),
    }


def c3_invariancias():
    # 1D espelho
    N, L = 300, 30.0
    faces = gerar_faces(L, N)
    x_c = centros(faces)
    zona = x_c < 12.0
    mat = {"sigma_t": np.ones(N), "sigma_s": np.where(zona, 0.95, 0.8),
           "nu_sigma_f": np.zeros(N), "fonte": np.where(zona, 1.0, 0.0)}
    A, q, _ = montar_1d(faces, mat, "vacuo", "reflexiva")
    phi = resolver_fonte_fixa(A, q).phi
    mat_e = {ch: v[::-1].copy() for ch, v in mat.items()}
    Ae, qe, _ = montar_1d(faces, mat_e, "reflexiva", "vacuo")
    phi_e = resolver_fonte_fixa(Ae, qe).phi
    dif_espelho = float(np.max(np.abs(phi - phi_e[::-1])) / phi.max())

    # 2D transposição (rotação de 90 graus do problema)
    Nx, Ny = 50, 70
    fx, fy = gerar_faces(20.0, Nx), gerar_faces(28.0, Ny)
    X, Y = np.meshgrid(centros(fx), centros(fy))
    dentro = (X < 8) & (Y < 12)
    mat2 = {"sigma_t": np.ones_like(X), "sigma_s": np.where(dentro, 0.99, 0.8),
            "nu_sigma_f": np.zeros_like(X), "fonte": np.where(dentro, 1.0, 0.0)}
    A2, q2, _ = montar_2d(fx, fy, mat2, "vacuo", "reflexiva", "vacuo", "reflexiva")
    p2 = resolver_fonte_fixa(A2, q2).phi.reshape(Ny, Nx)
    mat2t = {ch: v.T.copy() for ch, v in mat2.items()}
    A2t, q2t, _ = montar_2d(fy, fx, mat2t, "vacuo", "reflexiva", "vacuo", "reflexiva")
    p2t = resolver_fonte_fixa(A2t, q2t).phi.reshape(Nx, Ny)
    dif_transp = float(np.max(np.abs(p2 - p2t.T)) / p2.max())

    R["C3_invariancias"] = {
        "dif_rel_espelho_1d": dif_espelho,
        "dif_rel_transposicao_2d": dif_transp,
        "aprovado": bool(dif_espelho < 1e-11 and dif_transp < 1e-11),
    }


def c4_semente():
    faces, mat = _reator_refletido()
    A, _, F = montar_1d(faces, mat, "vacuo", "vacuo")
    ks = [float(resolver_autovalor(A, F, semente=s).k_eff)
          for s in (1, 7, 42, 123, 999)]
    spread = max(ks) - min(ks)
    R["C4_semente"] = {"sementes": [1, 7, 42, 123, 999], "k_eff": ks,
                       "dispersao": float(spread),
                       "aprovado": bool(spread < 1e-7)}


def c5_subcritico_profundo():
    """k ~ 0.3: estressa os shifts fixos do Wielandt."""
    N = 400
    faces = gerar_faces(20.0, N)
    mat = _mat(N, 1.0, 0.70, nsf=0.12)      # k_inf = 0.4
    A, _, F = montar_1d(faces, mat, "vacuo", "vacuo")
    saida = {}
    for metodo in ("wielandt", "power"):
        try:
            r = resolver_autovalor(A, F, metodo=metodo)
            saida[metodo] = {"k_eff": float(r.k_eff), "iteracoes": r.iteracoes,
                             "convergiu": True}
        except RuntimeError as exc:
            saida[metodo] = {"convergiu": False, "erro": str(exc)}
    ok = all(v.get("convergiu") for v in saida.values())
    if ok:
        saida["dif_k"] = abs(saida["wielandt"]["k_eff"] - saida["power"]["k_eff"])
        ok = saida["dif_k"] < 5e-7
    R["C5_subcritico"] = {**saida, "aprovado": bool(ok)}


def c6_malha_minima():
    """Nx = 3 (mínimo aceito): roda e o balanço fecha?"""
    N = 3
    faces = gerar_faces(20.0, N)
    mat = _mat(N, 1.0, 0.8, q=1.0)
    A, q, _ = montar_1d(faces, mat, "vacuo", "vacuo")
    phi = resolver_fonte_fixa(A, q).phi
    bal = balanco_1d(faces, mat, phi, "vacuo", "vacuo")
    R["C6_malha_minima"] = {"residuo_balanco": float(bal.residuo_relativo),
                            "aprovado": bool(bal.residuo_relativo < 1e-12)}


# ============================================================= D. ROBUSTEZ
def _rodar_cli(conteudo: str, nome: str, sufixo=".yaml"):
    arq = DADOS / f"input_{nome}{sufixo}"
    arq.write_text(textwrap.dedent(conteudo), encoding="utf-8")
    p = subprocess.run([PYTHON, str(RAIZ / "main.py"), str(arq)],
                       capture_output=True, text=True, cwd=RAIZ, timeout=120)
    tem_traceback = "Traceback" in p.stderr
    erro_limpo = "[ERRO de input]" in p.stderr or "[ERRO de input]" in p.stdout
    return {"returncode": p.returncode, "traceback_vazou": tem_traceback,
            "erro_limpo": erro_limpo,
            "stderr_resumo": p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""}


def d_robustez():
    base = """
        dimensao: 1
        malha: {Lx: 10.0, Nx: 50}
        bc: {esquerda: vacuo, direita: vacuo}
        regioes:
          - {x: [0.0, 10.0], sigma_t: 1.0, sigma_s: SIGMA_S, fonte: 1.0}
    """
    testes = {}

    r = _rodar_cli(base.replace("SIGMA_S", "1.5"), "sigma_invalido")
    testes["D1_sigma_s_maior"] = {**r, "esperado": "erro limpo",
                                  "aprovado": r["erro_limpo"] and not r["traceback_vazou"]}

    r = _rodar_cli("""
        dimensao: 1
        malha: {Lx: 10.0, Nx: 50}
        bc: {esquerda: vacuo, direita: vacuo}
        regioes:
          - {x: [0.0, 4.0], sigma_t: 1.0, sigma_s: 0.5, fonte: 1.0}
    """, "buraco")
    testes["D2_buraco_dominio"] = {**r, "esperado": "erro limpo",
                                   "aprovado": not r["traceback_vazou"] and r["returncode"] != 0}

    r = _rodar_cli('{"dimensao": 1, "malha": {', "json_malformado", sufixo=".json")
    testes["D3_json_malformado"] = {**r, "esperado": "erro limpo",
                                    "aprovado": not r["traceback_vazou"] and r["returncode"] != 0}

    r = _rodar_cli(base.replace("SIGMA_S", "0.5")
                       .replace("Nx: 50", 'Nx: "quatrocentos"'), "nx_string")
    testes["D4_nx_string"] = {**r, "esperado": "erro limpo",
                              "aprovado": not r["traceback_vazou"] and r["returncode"] != 0}

    arq = DADOS / "input_fonte_autovalor.yaml"
    arq.write_text(textwrap.dedent("""
        dimensao: 1
        modo: autovalor
        malha: {Lx: 20.0, Nx: 100}
        bc: {esquerda: vacuo, direita: vacuo}
        regioes:
          - {x: [0.0, 20.0], sigma_t: 1.0, sigma_s: 0.7, nu_sigma_f: 0.39, fonte: 5.0}
        saida: {prefixo: relatorio/dados/saida_fonte_autovalor}
    """), encoding="utf-8")
    p = subprocess.run([PYTHON, str(RAIZ / "main.py"), str(arq)],
                       capture_output=True, text=True, cwd=RAIZ, timeout=120)
    avisou = "fonte" in p.stderr.lower() and "ignor" in (p.stdout + p.stderr).lower()
    testes["D5_fonte_em_autovalor"] = {
        "returncode": p.returncode, "rodou": p.returncode == 0,
        "avisa_fonte_ignorada": avisou, "esperado": "aviso ao usuário",
        "aprovado": avisou,
    }

    r = _rodar_cli("", "vazio")
    testes["D6_yaml_vazio"] = {**r, "esperado": "erro limpo",
                               "aprovado": r["erro_limpo"] and not r["traceback_vazou"]}

    testes["D7_interface_no_centroide"] = _interface_centroide()
    R["D_robustez"] = testes


def _interface_centroide():
    """Interface em x=4.5 coincide com centróide (N=10, L=10). Ordem importa?"""
    import contextlib
    import io

    N, L = 10, 10.0
    faces = gerar_faces(L, N)
    x_c = centros(faces)
    regA = {"x": [0.0, 4.5], "sigma_t": 1.0, "sigma_s": 0.99,
            "nu_sigma_f": 0.0, "fonte": 1.0}
    regB = {"x": [4.5, 10.0], "sigma_t": 1.0, "sigma_s": 0.5,
            "nu_sigma_f": 0.0, "fonte": 0.0}
    from difusao.malha import materiais_1d
    phis = []
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        for ordem in ([regA, regB], [regB, regA]):
            mat = materiais_1d(ordem, x_c)
            A, q, _ = montar_1d(faces, mat, "vacuo", "vacuo")
            phis.append(resolver_fonte_fixa(A, q).phi)
    dif = float(np.max(np.abs(phis[0] - phis[1])) / phis[0].max())
    avisou = "[AVISO]" in err.getvalue() and "centróide" in err.getvalue()
    return {"dif_rel_entre_ordens": dif,
            "esperado": "independência da ordem ou aviso",
            "silencioso": not avisou,
            "aviso_emitido": avisou,
            "aprovado": bool(dif < 1e-12 or avisou)}


# ==================================================================== main
def main():
    passos = [a1_meio_infinito, a2_convergencia, a3_placa_critica,
              b1_continuidade_corrente, b2_sensibilidade, b3_efeito_refletor,
              b4_dominancia, b5_limite_difusivo,
              c1_pior_caso_combinado, c2_independencia_malha, c3_invariancias,
              c4_semente, c5_subcritico_profundo, c6_malha_minima,
              d_robustez]
    for fn in passos:
        print(f"[executando] {fn.__name__}")
        fn()

    arq = DADOS / "resultados.json"
    arq.write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nresultados salvos em {arq}")

    # resumo aprovado/reprovado
    print("\nRESUMO")
    for chave, valor in R.items():
        if chave == "D_robustez":
            for k2, v2 in valor.items():
                print(f"  {k2:<28} {'APROVADO' if v2.get('aprovado') else 'REPROVADO'}")
        elif isinstance(valor, dict) and "aprovado" in valor:
            print(f"  {chave:<28} {'APROVADO' if valor['aprovado'] else 'REPROVADO'}")
        else:
            print(f"  {chave:<28} (quantitativo)")


if __name__ == "__main__":
    main()
