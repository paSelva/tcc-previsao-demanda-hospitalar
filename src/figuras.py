"""
Geração das figuras do Capítulo 4.

Todas as figuras são desenhadas para **impressão em preto e branco**, que é como
o TCC é impresso e lido em papel. Isso impõe três regras:

    · séries são distinguidas por estilo de traço e marcador, nunca só por cor;
    · barras são distinguidas por textura (hachura), nunca só por preenchimento;
    · o contraste é verificado em escala de cinza, não em tela colorida.

Saída em `outputs/figuras/`, 300 dpi, largura compatível com página A4 com
margens ABNT (aproximadamente 15 cm de área útil).

Uso:
    ./venv/bin/python src/figuras.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parent.parent
PROCESSADO = RAIZ / "data" / "processed"
TABELAS = RAIZ / "outputs" / "tabelas"
FIGURAS = RAIZ / "outputs" / "figuras"

LARGURA = 5.9          # polegadas ≈ 15 cm de área útil da página
CINZAS = ["#1a1a1a", "#5c5c5c", "#8f8f8f", "#c2c2c2"]
HACHURAS = ["", "///", "...", "xxx"]
TRACOS = ["-", "--", "-.", ":"]
MARCADORES = ["o", "s", "^", "D"]

ROTULO = {"consulta": "Consultas ambulatoriais", "internacao": "Internações hospitalares"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#4d4d4d",
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

milhar = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))


def salva(fig, nome: str) -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    caminho = FIGURAS / f"{nome}.png"
    fig.savefig(caminho)
    plt.close(fig)
    print(f"  {caminho.relative_to(RAIZ)}")


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------

atend = pd.read_csv(PROCESSADO / "atendimentos.csv.gz", parse_dates=["data_hora"])
atend["periodo"] = atend["data_hora"].dt.to_period("M").dt.to_timestamp()
painel = pd.read_csv(PROCESSADO / "painel.csv.gz", parse_dates=["periodo"])
# Figuras 6 a 8 leem as tabelas de `validacao.py`, as mesmas das Tabelas 11, 14
# e 15 do texto — figura e tabela vizinhas precisam mostrar os mesmos números.
agreg = pd.read_csv(TABELAS / "validacao_geral.csv")
estrato = pd.read_csv(TABELAS / "validacao_estrato.csv")
importancia = pd.read_csv(TABELAS / "validacao_importancia.csv")
importancia = importancia[importancia["variavel"] != "historico (lags + medias)"]
NOME = {"Baseline": "Persistência", "Sazonal 12m": "Sazonal 12 m"}
REFERENCIAS = {"Baseline", "Sazonal 12m"}
previsoes = pd.read_csv(TABELAS / "previsoes.csv", parse_dates=["periodo"])

serie = atend.groupby(["tipo", "periodo"]).size().unstack(0).sort_index()


# --------------------------------------------------------------------------
# Figura 1 — séries mensais
# --------------------------------------------------------------------------

def figura_1() -> None:
    fig, eixos = plt.subplots(2, 1, figsize=(LARGURA, 4.4), sharex=True)
    for ax, tipo in zip(eixos, ["consulta", "internacao"]):
        s = serie[tipo]
        ax.plot(s.index, s.values, color=CINZAS[0], linewidth=1.4)
        ax.fill_between(s.index, 0, s.values, color=CINZAS[3], alpha=0.35)
        ax.set_ylim(0, s.max() * 1.12)
        ax.set_title(ROTULO[tipo], loc="left")
        ax.set_ylabel("Atendimentos/mês")
        ax.yaxis.set_major_formatter(milhar)
    eixos[1].set_xlabel("Mês de referência")
    salva(fig, "fig01-series-mensais")


# --------------------------------------------------------------------------
# Figura 2 — sazonalidade
# --------------------------------------------------------------------------

def figura_2() -> None:
    sz = atend.groupby(["tipo", atend["periodo"].dt.month]).size().unstack(0)
    sz = sz / sz.mean()
    fig, ax = plt.subplots(figsize=(LARGURA, 2.7))
    for i, tipo in enumerate(["consulta", "internacao"]):
        ax.plot(sz.index, sz[tipo], linestyle=TRACOS[i], marker=MARCADORES[i],
                markersize=4, color=CINZAS[i * 2], linewidth=1.3, label=ROTULO[tipo])
    ax.axhline(1.0, color=CINZAS[2], linewidth=0.8, linestyle=":")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                        "Jul", "Ago", "Set", "Out", "Nov", "Dez"])
    ax.set_ylabel("Índice sazonal")
    ax.set_xlabel("Mês")
    ax.legend(loc="lower left", ncol=2)
    salva(fig, "fig02-sazonalidade")


# --------------------------------------------------------------------------
# Figura 3 — esparsidade dos recortes candidatos
# --------------------------------------------------------------------------

def figura_3() -> None:
    a = atend.loc[atend["idade"].notna()].copy()
    a["faixa"] = pd.cut(a["idade"], bins=[-1, 14, 29, 44, 59, 74, 200],
                        labels=["0-14", "15-29", "30-44", "45-59", "60-74", "75+"])
    a["sexo3"] = a["sexo"].where(a["sexo"].isin(["FEMININO", "MASCULINO"]), "OUTRO")
    top20 = a["municipio"].value_counts().head(20).index
    a["mun20"] = a["municipio"].where(a["municipio"].isin(top20), "OUTROS")

    recortes = [
        ("esp.", ["especialidade"]),
        ("esp. × faixa", ["especialidade", "faixa"]),
        ("esp. × faixa × sexo", ["especialidade", "faixa", "sexo3"]),
        ("esp. × munic.", ["especialidade", "mun20"]),
        ("esp. × munic. × faixa × sexo", ["especialidade", "mun20", "faixa", "sexo3"]),
    ]
    nper = a["periodo"].nunique()
    dados = {}
    for tipo in ["consulta", "internacao"]:
        sub = a[a["tipo"] == tipo]
        zeros = []
        for _, chave in recortes:
            cont = sub.groupby(["periodo", *chave], observed=True).size()
            total = nper * int(np.prod([sub[c].nunique() for c in chave]))
            zeros.append(100 * (1 - len(cont) / total))
        dados[tipo] = zeros

    y = np.arange(len(recortes))
    fig, ax = plt.subplots(figsize=(LARGURA, 2.9))
    alt = 0.38
    for i, tipo in enumerate(["consulta", "internacao"]):
        ax.barh(y + (0.5 - i) * alt, dados[tipo], height=alt,
                color=CINZAS[i * 2 + 1], edgecolor=CINZAS[0], linewidth=0.7,
                hatch=HACHURAS[i * 2], label=ROTULO[tipo])
    ax.axvline(90, color=CINZAS[0], linestyle="--", linewidth=1.0)
    ax.text(88.5, 0.15, "limiar prático\nde 90%", fontsize=6.8, color=CINZAS[0],
            ha="right", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in recortes])
    ax.set_xlabel("Células vazias na grade completa (%)")
    ax.set_xlim(0, 102)
    ax.invert_yaxis()
    # Legenda abaixo do eixo: as barras chegam perto de 100% e não deixam
    # espaço interno livre em nenhum canto.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    ax.grid(axis="y", visible=False)
    salva(fig, "fig03-esparsidade")


# --------------------------------------------------------------------------
# Figura 4 — distribuição do alvo
# --------------------------------------------------------------------------

def figura_4() -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA, 2.5))
    for ax, tipo in zip(eixos, ["consulta", "internacao"]):
        v = painel.loc[painel["tipo"] == tipo, "n"]
        ax.hist(np.log1p(v), bins=34, color=CINZAS[2],
                edgecolor=CINZAS[0], linewidth=0.6)
        pct = 100 * (v == 0).mean()
        ax.axvline(0, color=CINZAS[0], linestyle="--", linewidth=1.1)
        ax.text(0.22, ax.get_ylim()[1] * 0.86, f"{pct:.1f}%\nzeros", fontsize=7.5)
        ax.set_title(ROTULO[tipo], loc="left")
        ax.set_xlabel("log(1 + atendimentos)")
    eixos[0].set_ylabel("Células do painel")
    salva(fig, "fig04-distribuicao-alvo")


# --------------------------------------------------------------------------
# Figura 5 — especialidades com maior demanda
# --------------------------------------------------------------------------

def figura_5() -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA, 3.2))
    for ax, tipo in zip(eixos, ["consulta", "internacao"]):
        v = (atend[atend["tipo"] == tipo]["especialidade"]
             .value_counts().head(10).sort_values())
        ax.barh([e.title()[:24] for e in v.index], v.values,
                color=CINZAS[2], edgecolor=CINZAS[0], linewidth=0.7)
        ax.set_title(ROTULO[tipo], loc="left")
        ax.set_xlabel("Atendimentos (2023–2026)")
        ax.xaxis.set_major_formatter(milhar)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=6.8)
    salva(fig, "fig05-top-especialidades")


# --------------------------------------------------------------------------
# Figura 6 — MAE por modelo (reagregado)
# --------------------------------------------------------------------------

def figura_6() -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA, 2.7))
    for ax, tipo in zip(eixos, ["consulta", "internacao"]):
        g = agreg[agreg["tipo"] == tipo].sort_values("MAE")
        base = g.loc[g["modelo"] == "Baseline", "MAE"].iloc[0]
        cores = [CINZAS[3] if m in REFERENCIAS else CINZAS[1] for m in g["modelo"]]
        hach = ["xxx" if m in REFERENCIAS else "" for m in g["modelo"]]
        barras = ax.bar(range(len(g)), g["MAE"], color=cores,
                        edgecolor=CINZAS[0], linewidth=0.7)
        for b, h in zip(barras, hach):
            b.set_hatch(h)
        ax.axhline(base, color=CINZAS[0], linestyle="--", linewidth=1.0)
        ax.set_xticks(range(len(g)))
        quebra = {"Random Forest": "Random\nForest", "Gradient Boosting": "Gradient\nBoosting",
                  "Baseline": "Persis-\ntência", "Sazonal 12m": "Sazonal\n12 m"}
        ax.set_xticklabels([quebra.get(m, m) for m in g["modelo"]], fontsize=6.2)
        ax.set_title(ROTULO[tipo], loc="left")
        ax.set_ylabel("MAE" if tipo == "consulta" else "")
        for i, v in enumerate(g["MAE"]):
            ax.text(i, v * 1.02, f"{v:.2f}".replace(".", ","), ha="center", fontsize=6.4)
        ax.set_ylim(0, g["MAE"].max() * 1.18)
        ax.grid(axis="x", visible=False)
    salva(fig, "fig06-mae-modelos")


# --------------------------------------------------------------------------
# Figura 7 — MAE por estrato de volume
# --------------------------------------------------------------------------

def figura_7() -> None:
    ordem_estrato = ["alto", "medio", "baixo"]
    modelos = ["Baseline", "Sazonal 12m", "Poisson", "Random Forest", "Gradient Boosting"]
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA, 2.9))
    for ax, tipo in zip(eixos, ["consulta", "internacao"]):
        g = estrato[estrato["tipo"] == tipo]
        x = np.arange(len(ordem_estrato))
        larg = 0.16
        for i, mod in enumerate(modelos):
            vals = [g[(g["estrato"] == e) & (g["modelo"] == mod)]["MAE"].iloc[0]
                    for e in ordem_estrato]
            # MAE normalizado pelo do baseline: escalas dos estratos diferem
            # em duas ordens de grandeza e não caberiam no mesmo eixo.
            base = [g[(g["estrato"] == e) & (g["modelo"] == "Baseline")]["MAE"].iloc[0]
                    for e in ordem_estrato]
            rel = [v / b for v, b in zip(vals, base)]
            ax.bar(x + (i - 2) * larg, rel, larg, label=NOME.get(mod, mod),
                   color=(CINZAS + ["#ffffff"])[i], edgecolor=CINZAS[0], linewidth=0.6,
                   hatch=(HACHURAS + ["\\\\\\"])[i])
        ax.axhline(1.0, color=CINZAS[0], linewidth=1.0, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(["Alto", "Médio", "Baixo"])
        ax.set_title(ROTULO[tipo], loc="left")
        ax.set_ylabel("MAE relativo à persistência" if tipo == "consulta" else "")
        ax.grid(axis="x", visible=False)
    # Legenda única acima dos dois painéis: dentro do eixo ela colidiria com a
    # barra alta do estrato baixo, que é justamente o achado da figura.
    manipuladores, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(manipuladores, rotulos, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.09), fontsize=7)
    salva(fig, "fig07-mae-estrato")


# --------------------------------------------------------------------------
# Figura 8 — importância por permutação
# --------------------------------------------------------------------------

def figura_8() -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA, 3.0))
    for ax, tipo in zip(eixos, ["consulta", "internacao"]):
        g = importancia[(importancia["tipo"] == tipo)
                        & (importancia["modelo"] == "Random Forest")]
        g = g.nlargest(8, "peso_%").sort_values("peso_%")
        ax.barh(g["variavel"], g["peso_%"],
                color=CINZAS[2], edgecolor=CINZAS[0], linewidth=0.7)
        ax.set_title(ROTULO[tipo], loc="left")
        ax.set_xlabel("Peso relativo (%)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=6.8)
    salva(fig, "fig08-importancia-permutacao")


# --------------------------------------------------------------------------
# Figura 9 — observado versus previsto no período de teste
# --------------------------------------------------------------------------

def figura_9() -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA, 2.7))
    for ax, tipo in zip(eixos, ["consulta", "internacao"]):
        p = previsoes[previsoes["tipo"] == tipo]
        obs = p[p["modelo"] == "Baseline"].groupby("periodo")["n"].sum()
        ax.plot(obs.index, obs.values, color=CINZAS[0], linewidth=1.5,
                marker="o", markersize=4, label="Observado")
        for i, mod in enumerate(["Random Forest", "Baseline"]):
            pr = p[p["modelo"] == mod].groupby("periodo")["predito"].sum()
            ax.plot(pr.index, pr.values, linestyle=TRACOS[i + 1],
                    marker=MARCADORES[i + 1], markersize=3.5,
                    color=CINZAS[i + 1], linewidth=1.2, label=mod)
        ax.set_title(ROTULO[tipo], loc="left")
        ax.set_ylabel("Atendimentos/mês" if tipo == "consulta" else "")
        ax.yaxis.set_major_formatter(milhar)
        ax.tick_params(axis="x", rotation=45, labelsize=6.5)
    eixos[0].legend(loc="lower left", fontsize=6.5)
    salva(fig, "fig09-observado-previsto")


if __name__ == "__main__":
    print("Gerando figuras do Capítulo 4:")
    for f in (figura_1, figura_2, figura_3, figura_4, figura_5,
              figura_6, figura_7, figura_8, figura_9):
        f()
    print(f"\n{len(list(FIGURAS.glob('*.png')))} figuras em outputs/figuras/")
