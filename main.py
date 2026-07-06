from __future__ import annotations

import argparse
import sys
import time

from difusao.balanco import balanco_1d, balanco_2d
from difusao.input_data import ErroDeInput, ler_input
from difusao.input_echo import eco_input
from difusao.malha import centros, gerar_faces, materiais_1d, materiais_2d
from difusao.montagem import montar_1d, montar_2d
from difusao.solver import resolver_autovalor, resolver_fonte_fixa
from difusao.output import escrever_saida
from difusao.version import imprime_versao

_LIMIAR_BALANCO = 1.0e-8


def executar_caso(caminho_input: str) -> int:
    dados = ler_input(caminho_input)
    eco_input(dados)

    malha = dados["malha"]
    bc = dados["bc"]
    t0 = time.time()

    if dados["dimensao"] == 1:
        faces = gerar_faces(malha["Lx"], malha["Nx"], malha["graduacao_x"])
        x_c = centros(faces)
        y_c = None
        mat = materiais_1d(dados["regioes"], x_c)
        A, q, F = montar_1d(faces, mat, bc["esquerda"], bc["direita"])
        bal_args = (faces, mat)
        bal_fn = lambda phi, k: balanco_1d(*bal_args, phi, bc["esquerda"],
                                           bc["direita"], k_eff=k)
    else:
        faces_x = gerar_faces(malha["Lx"], malha["Nx"], malha["graduacao_x"])
        faces_y = gerar_faces(malha["Ly"], malha["Ny"], malha["graduacao_y"])
        x_c, y_c = centros(faces_x), centros(faces_y)
        mat = materiais_2d(dados["regioes"], x_c, y_c)
        A, q, F = montar_2d(faces_x, faces_y, mat, bc["esquerda"],
                            bc["direita"], bc["inferior"], bc["superior"])
        bal_fn = lambda phi, k: balanco_2d(faces_x, faces_y, mat, phi,
                                           bc["esquerda"], bc["direita"],
                                           bc["inferior"], bc["superior"], k_eff=k)

    slv = dados["solver"]
    if dados["modo"] == "fonte_fixa":
        res = resolver_fonte_fixa(A, q, metodo=slv["metodo"], tol=slv["tol"])
        phi, k_eff, fatoracoes = res.phi, None, None
        info, iters = res.metodo, res.iteracoes
    else:
        av = slv["autovalor"]
        res = resolver_autovalor(A, F, metodo=av["metodo"],
                                 tol_k=av["tol_k"], tol_phi=av["tol_phi"])
        phi, k_eff, fatoracoes = res.phi, res.k_eff, res.fatoracoes
        info, iters = res.metodo, res.iteracoes

    tempo = time.time() - t0
    if dados["dimensao"] == 2:
        phi = phi.reshape(malha["Ny"], malha["Nx"])

    bal = bal_fn(phi, k_eff)

    print("RESULTADOS")
    if k_eff is not None:
        print(f"  k_eff = {k_eff:.8f}  ({info}, {iters} iterações, "
              f"{fatoracoes} fatorações LU)")
    else:
        print(f"  solver: {info}" + (f", {iters} iterações" if iters else ""))
    print(f"  phi: min = {phi.min():.6e}, max = {phi.max():.6e}")
    rotulo = "fissão/k" if k_eff is not None else "fonte"
    print(f"  balanço: {rotulo} = {bal.fonte:.6f} | absorção = {bal.absorcao:.6f} "
          f"| fuga = {bal.fuga:.6f}")
    print(f"  resíduo relativo do balanço = {bal.residuo_relativo:.3e}")
    print(f"  tempo de solução = {tempo*1000:.1f} ms")

    if bal.residuo_relativo > _LIMIAR_BALANCO:
        print(f"  [ERRO] balanço acima do limiar ({_LIMIAR_BALANCO:.0e}): "
              "resultado NÃO confiável.", file=sys.stderr)
        return 2

    arq_resumo, arq_fluxo = escrever_saida(
        dados, x_c, y_c, phi, bal, info, iters,
        k_eff=k_eff, fatoracoes=fatoracoes)
    print(f"  saídas: {arq_resumo} | {arq_fluxo}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Solver de difusão de nêutrons monoenergética 1D/2D "
                    "(Opções 1 e 3 do projeto final).")
    parser.add_argument("inputs", nargs="+",
                        help="um ou mais arquivos de entrada (.yaml/.json)")
    args = parser.parse_args(argv)

    imprime_versao()
    falhas = 0
    for caminho in args.inputs:
        print(f"\n>>> caso: {caminho}")
        try:
            falhas += executar_caso(caminho) != 0
        except ErroDeInput as exc:
            print(f"  [ERRO de input] {exc}", file=sys.stderr)
            falhas += 1
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
