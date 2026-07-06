# difusao — Solver de Difusão de Nêutrons 1D/2D (Opções 1 e 3)

Projeto final de Física de Reatores: solver da equação de difusão de
nêutrons monoenergética, estacionária, em meios heterogêneos.

* **Opção 3** — 1D, fonte fixa **e** problema de autovalor ($k_{\text{eff}}$);
* **Opção 1** — 2D, com vácuo nas faces inferior/esquerda e reflexão nas
  faces superior/direita (qualquer combinação de BCs é aceita).

## Status

Funcional e validado. Suíte de testes: **17/17 passando** (ver seção
*Testes*). Limitação física conhecida: a aproximação de difusão degrada
perto de fronteiras de vácuo e interfaces quando $c = \Sigma_s/\Sigma_t \to 1$
— é limitação do modelo, não defeito do código.

## Método

* Volumes finitos *cell-centered*; estêncil de 3 pontos (1D) / 5 pontos (2D).
* Interfaces de material por **média harmônica** generalizada do coeficiente
  de difusão (suporta malha não uniforme).
* Vácuo pela condição de **Marshak** ($\phi/4 - (D/2)\,\partial\phi/\partial n = 0$),
  implementada como condutância de borda $\alpha = 2D/(h+4D)$; reflexão =
  corrente nula.
* Sistemas lineares: **LU esparsa** (`splu`, ordenação MMD) fatorada uma única
  vez e reutilizada — ou CG + Jacobi (`solver.metodo: cg`).
* Autovalor: power iteration com **shift de Wielandt em dois estágios**
  (padrão) ou power iteration clássica, com critério duplo de parada
  (em $k$ e na forma do fluxo).
* **Balanço de nêutrons** (fonte = absorção + fuga) verificado em toda
  execução; resíduo acima de 1e-8 aborta com erro.

A discretização foi derivada simbolicamente (SymPy) nos notebooks que
acompanham o projeto: `difusao_opcoes_1_e_3.ipynb` (derivações e validação)
e o estudo de otimização (fatorações LU/QR, Wielandt, simetria, malha
graduada) que fundamenta as escolhas de solver acima.

## Estrutura (conforme exigido no enunciado)

```
main.py                  módulo principal (CLI)
plotar.py                geração de figuras (Seaborn + LaTeX)
difusao/
  version.py             subrotina de versão
  input_data.py          subrotina de leitura do input (YAML/JSON) + validação
  input_echo.py          subrotina de eco do input
  malha.py               malha (uniforme/graduada) e mapeamento de materiais
  montagem.py            montagem das matrizes (volumes finitos)
  solver.py              fonte fixa (LU/CG) e autovalor (power/Wielandt)
  balanco.py             diagnóstico de conservação de nêutrons
  output.py              escrita de resultados (resumo .txt + fluxo .csv)
  estilo.py              estilo gráfico (Seaborn + usetex c/ fallback)
inputs/                  10 casos obrigatórios + exemplos 2D e autovalor
outputs/                 amostras de saída geradas pelos inputs acima
tests/test_validacao.py  suíte pytest
```

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy, scipy, pyyaml (+pytest p/ testes)
```

## Uso

```bash
# um caso
python main.py inputs/caso01.yaml

# todos os 10 casos obrigatórios do enunciado
python main.py inputs/caso*.yaml

# 2D (Opção 1) e autovalor com Wielandt
python main.py inputs/exemplo_2d.yaml inputs/exemplo_autovalor.yaml
```

Cada execução imprime: banner de versão → eco do input → resultados →
balanço de nêutrons, e grava `<prefixo>_resumo.txt` e `<prefixo>_fluxo.csv`.

## Gráficos (Seaborn + fontes LaTeX)

```bash
# figuras individuais (PNG 300 dpi + PDF vetorial) e a combinada dos 10 casos
python plotar.py inputs/caso*.yaml --combinar
python plotar.py inputs/exemplo_2d.yaml inputs/exemplo_autovalor.yaml
```

* 1D → perfil $\phi(x)$ com regiões de material sombreadas e interfaces marcadas;
* 2D → mapa de calor $\phi(x,y)$ com curvas de nível + cortes horizontais;
* autovalor → perfil com $k_\text{eff}$ anotado no título;
* `--combinar` → painéis dos casos 1D agrupados pelo material 2 (escala log);
* `--contexto paper|notebook|talk|poster` ajusta tamanhos (contextos do Seaborn).

**Fontes LaTeX.** O estilo (`difusao/estilo.py`) detecta automaticamente uma
instalação LaTeX e ativa `text.usetex=True` — todo o texto das figuras é
composto pelo LaTeX (Computer Modern). Para habilitar no Ubuntu/Debian:

```bash
sudo apt install texlive-latex-base texlive-fonts-recommended cm-super-minimal dvipng
```

Sem LaTeX instalado (ou com `--sem-latex`), o script cai automaticamente no
*mathtext* do matplotlib com `fontset='cm'` — mesma família Computer Modern,
sem nenhuma dependência externa. As figuras de amostra em `outputs/` foram
geradas no modo `usetex`.

### Formato do input

Ver cabeçalho de `difusao/input_data.py` (documentação completa de todos
os campos, obrigatórios e defaults) ou os exemplos em `inputs/`. Resumo:

```yaml
dimensao: 1            # 1 ou 2
modo: fonte_fixa       # fonte_fixa | autovalor
malha:   {Lx: 20.0, Nx: 400}            # 2D: + Ly, Ny; graduacao_x p/ malha graduada
bc:      {esquerda: vacuo, direita: vacuo}   # 2D: + inferior, superior
regioes:                                # a última região listada sobrepõe
  - {x: [0.0, 10.0], sigma_t: 1.0, sigma_s: 0.8, fonte: 1.0}
  - {x: [10.0, 20.0], sigma_t: 1.0, sigma_s: 0.8, fonte: 0.0}
solver:  {metodo: direto}               # direto | cg; autovalor: power | wielandt
saida:   {prefixo: outputs/meucaso}
```

## Os 10 problemas-teste obrigatórios

`inputs/caso01.yaml` … `caso10.yaml`: placa de 20 cm, metade material 1
($\Sigma_{t}=1$, $Q=1$, $\Sigma_{s1}\in\{0.8, 0.9, 0.99, 0.999, 0.9999\}$),
metade material 2 ($\Sigma_t=1$, $Q=0$, $\Sigma_{s2}\in\{0.8, 0.99\}$), vácuo
em ambos os lados. Resíduo de balanço obtido: $10^{-14}$–$10^{-13}$ em todos.

## Testes

```bash
python -m pytest tests/ -v
```

Cobertura: meio infinito analítico (erro ~1e-14), convergência espacial de
ordem 2, balanço dos 10 casos obrigatórios, validação cruzada 1D↔2D,
equivalência Wielandt ≡ power (com Wielandt ≥3× mais rápido), equivalência
¼-de-domínio ↔ domínio espelhado completo, e detecção de inputs inválidos.

## Desempenho (medido)

| Situação | Técnica usada | Ganho |
|---|---|---|
| Sistemas lineares repetidos | LU fatorada 1×, reutilizada | ~12× vs refatorar |
| Fill-in da LU 2D | ordenação MMD_AT_PLUS_A | fill 4× menor que natural |
| Autovalor com núcleos fracamente acoplados | Wielandt 2 estágios | até ~35× menos iterações |
| Problemas simétricos | BCs reflexivas = ¼ de domínio | ~4× tempo |
| Camada-limite no vácuo | `graduacao_x > 1` | erro ~3× menor, mesmo N |
