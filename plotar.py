from __future__ import annotations

import argparse
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from difusao.estilo import PALETA, aplicar_estilo
from difusao.input_data import ler_input


def _carregar(dados: dict):
    prefixo = pathlib.Path(dados["saida"]["prefixo"])
    csv = pathlib.Path(f"{prefixo}_fluxo.csv")
    if not csv.exists():
        sys.exit(f"[erro] {csv} não existe — rode antes: python main.py {dados['_arquivo']}")
    tabela = np.genfromtxt(csv, delimiter=",", names=True)

    k_eff = None
    resumo = pathlib.Path(f"{prefixo}_resumo.txt")
    if resumo.exists():
        m = re.search(r"k_eff\s*=\s*([0-9.]+)", resumo.read_text(encoding="utf-8"))
        if m:
            k_eff = float(m.group(1))
    return tabela, k_eff


def _rotulo_regiao(reg: dict) -> str:
    return (rf"{reg['nome']}: $\Sigma_t={reg['sigma_t']:g}$, "
            rf"$\Sigma_s={reg['sigma_s']:g}$, $Q={reg['fonte']:g}$")


def plotar_1d(dados: dict, tabela, k_eff, arq_base: pathlib.Path) -> None:
    x, phi = tabela["x"], tabela["phi"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, phi, lw=1.8, color=sns.color_palette(PALETA, 5)[3])

    bordas = sorted({v for reg in dados["regioes"] for v in reg["x"]})
    for x_int in bordas[1:-1]:
        ax.axvline(x_int, color="k", ls=":", lw=0.8, alpha=0.6)
    for i, reg in enumerate(dados["regioes"]):
        ax.axvspan(*reg["x"], alpha=0.06 + 0.05 * (i % 2), color="gray",
                   label=_rotulo_regiao(reg))

    ax.set_xlabel(r"$x$ [cm]")
    ax.set_ylabel(r"$\phi(x)$ [cm$^{-2}\,$s$^{-1}$]")
    titulo = dados["titulo"]
    if k_eff is not None:
        titulo += rf"  ($k_\mathrm{{eff}} = {k_eff:.6f}$)"
    ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(arq_base.with_suffix(f".{ext}"))
    plt.close(fig)


def plotar_2d(dados: dict, tabela, arq_base: pathlib.Path) -> None:
    Nx, Ny = dados["malha"]["Nx"], dados["malha"]["Ny"]
    X = tabela["x"].reshape(Ny, Nx)
    Y = tabela["y"].reshape(Ny, Nx)
    P = tabela["phi"].reshape(Ny, Nx)

    fig = plt.figure(figsize=(10.5, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.pcolormesh(X, Y, P, cmap="rocket", shading="nearest")
    ax0.contour(X, Y, P, levels=10, colors="white", linewidths=0.5, alpha=0.6)
    for reg in dados["regioes"][1:]:
        x0, x1 = reg["x"]; y0, y1 = reg["y"]
        ax0.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                 color="cyan", lw=0.8, ls=":")
    ax0.set_aspect("equal")
    ax0.set_xlabel(r"$x$ [cm]"); ax0.set_ylabel(r"$y$ [cm]")
    ax0.set_title(r"$\phi(x,y)$", fontsize=10)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04,
                 label=r"$\phi$ [cm$^{-2}\,$s$^{-1}$]")

    ax1 = fig.add_subplot(gs[0, 1])
    cores = sns.color_palette(PALETA, 4)
    Ly = dados["malha"]["Ly"]
    for cor, frac in zip(cores, (0.25, 0.5, 0.75, 1.0)):
        j = min(int(frac * Ny) - 1, Ny - 1)
        ax1.plot(X[j], P[j], lw=1.5, color=cor,
                 label=rf"$y = {frac * Ly:g}$ cm")
    ax1.set_xlabel(r"$x$ [cm]")
    ax1.set_ylabel(r"$\phi(x, y)$")
    ax1.set_title("cortes horizontais", fontsize=10)
    ax1.legend(fontsize=8)

    fig.suptitle(dados["titulo"], fontsize=11, y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(arq_base.with_suffix(f".{ext}"))
    plt.close(fig)


def plotar_combinada(lista_dados: list[dict], arq_base: pathlib.Path) -> None:
    casos = []
    for dados in lista_dados:
        if (dados["dimensao"] != 1 or len(dados["regioes"]) < 2
                or dados["modo"] != "fonte_fixa"):
            continue
        tabela, _ = _carregar(dados)
        casos.append((dados["regioes"][0]["sigma_s"],
                      dados["regioes"][1]["sigma_s"], tabela))
    if not casos:
        return

    grupos = sorted({c[1] for c in casos})
    s1_unicos = sorted({c[0] for c in casos})
    cores = dict(zip(s1_unicos, sns.color_palette(PALETA, len(s1_unicos))))

    fig, axes = plt.subplots(len(grupos), 1, figsize=(7.5, 3.4 * len(grupos)),
                             sharex=True, squeeze=False)
    for ax, s2 in zip(axes[:, 0], grupos):
        for s1, s2c, tab in sorted(casos):
            if s2c != s2:
                continue
            ax.plot(tab["x"], tab["phi"], lw=1.6, color=cores[s1],
                    label=rf"$\Sigma_{{s1}} = {s1}$")
        ax.axvline(10.0, color="k", ls=":", lw=0.8, alpha=0.6)
        ax.set_yscale("log")
        ax.set_ylabel(r"$\phi(x)$ [cm$^{-2}\,$s$^{-1}$]")
        ax.set_title(rf"material 2: $\Sigma_{{s2}} = {s2}$"
                     "  (pontilhado = interface)", fontsize=10)
        ax.legend(fontsize=8, ncols=2)
    axes[-1, 0].set_xlabel(r"$x$ [cm]")
    fig.suptitle("Problemas-teste obrigatórios — fluxo escalar", y=1.005)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(arq_base.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"  figura combinada: {arq_base}.png/.pdf")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="plotar.py",
        description="Gera figuras (Seaborn + LaTeX) dos casos já executados.")
    parser.add_argument("inputs", nargs="+", help="arquivos de input .yaml")
    parser.add_argument("--combinar", action="store_true",
                        help="figura única sobrepondo os casos 1D")
    parser.add_argument("--sem-latex", action="store_true",
                        help="força mathtext (não exige LaTeX instalado)")
    parser.add_argument("--contexto", default="notebook",
                        choices=["paper", "notebook", "talk", "poster"])
    args = parser.parse_args(argv)

    usetex = aplicar_estilo(usetex=False if args.sem_latex else None,
                            contexto=args.contexto)
    print(f"fontes: {'LaTeX (usetex)' if usetex else 'mathtext Computer Modern'}")

    lista_dados = []
    for caminho in args.inputs:
        dados = ler_input(caminho)
        lista_dados.append(dados)
        tabela, k_eff = _carregar(dados)
        arq_base = pathlib.Path(dados["saida"]["prefixo"] + "_grafico")
        if dados["dimensao"] == 1:
            plotar_1d(dados, tabela, k_eff, arq_base)
        else:
            plotar_2d(dados, tabela, arq_base)
        print(f"  {caminho} -> {arq_base}.png/.pdf")

    if args.combinar:
        plotar_combinada(lista_dados, pathlib.Path("outputs/combinado_grafico"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
