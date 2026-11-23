"""
Validação complementar dos modelos (CRISP-DM — Fase 5, Avaliação).

Responde às pendências levantadas na revisão de 16/11/2026 sem alterar o
experimento principal de `modelagem.py`: os modelos, as origens e as previsões
são os mesmos. Este script LÊ `outputs/tabelas/previsoes.csv` e o painel, e
acrescenta as evidências pedidas.

    (a) Premissas da Regressão de Poisson
        - proporção de zeros por base e por estrato;
        - razão variância/média por estrato;
        - estatística de Pearson χ²/gl do GLM ajustado no treino;
        - teste de sobredispersão de Cameron e Trivedi (1990);
        - zeros observados × zeros esperados pelo Poisson (excesso de zeros).
    (c) Resultado de CADA origem (jan–jun/2026): MAE, RMSE e ganho sobre a
        persistência, por modelo e por base.
    (d) Baseline sazonal de 12 meses (valor do mesmo grupo no mesmo mês do ano
        anterior), comparação pareada entre modelos com intervalo de confiança
        por bootstrap e erro relativo (WAPE) por especialidade.
    (e) Importância por permutação com 10 repetições por origem, desvio-padrão,
        valores negativos preservados e permutação conjunta do bloco de
        variáveis de histórico (defasagens + médias móveis), que são
        fortemente correlacionadas entre si.
    (f) Métricas com quatro casas decimais, para separar empate real de
        arredondamento.

Como as métricas são calculadas (unidade de avaliação)
------------------------------------------------------
O modelo prevê ŷ para cada célula c = (mês m, especialidade s, faixa f).

  Nível da decisão (Tabelas 10 e 13):
      Y(m,s) = Σ_f y(m,s,f)          Ŷ(m,s) = Σ_f ŷ(m,s,f)
      MAE    = (1/N) Σ_(m,s) |Y(m,s) − Ŷ(m,s)|,   N = 6 meses × nº de especialidades
  As previsões NÃO são arredondadas: somam-se valores contínuos e o erro é
  medido sobre a soma. Arredondar cada célula antes de somar acumularia erro
  de arredondamento nas especialidades com muitas faixas.

  Nível do estrato (Tabela 11):
      MAE_estrato = (1/N_k) Σ_(c ∈ estrato k) |y(c) − ŷ(c)|
  calculado na CÉLULA, porque o estrato é atributo da célula (especialidade ×
  faixa), definido pela média da célula no treino. Por isso a Tabela 11 não
  pode ser comparada em valor absoluto com a Tabela 10: as unidades diferem.

Uso:
    ./venv/bin/python src/modelagem.py      # gera previsoes.csv (se ainda não existir)
    ./venv/bin/python src/validacao.py            # completo (~15 min)
    ./venv/bin/python src/validacao.py --rapido   # sem a importância por permutação
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from modelagem import ajusta_poisson, matriz, origem_da_coluna, preve

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent.parent
PROCESSADO = RAIZ / "data" / "processed"
TABELAS = RAIZ / "outputs" / "tabelas"

BASELINE = "Baseline"
SAZONAL = "Sazonal 12m"
ORDEM = [BASELINE, SAZONAL, "Poisson", "Random Forest", "Gradient Boosting"]
BOOTSTRAP = 5000
REPETICOES_PERMUTACAO = 10


# --------------------------------------------------------------------------
# (d) baseline sazonal
# --------------------------------------------------------------------------
def acrescenta_sazonal(prev: pd.DataFrame, painel: pd.DataFrame) -> pd.DataFrame:
    """Previsão sazonal ingênua: y(m) ≈ y(m − 12) no mesmo grupo.

    `lag_12` já existe no painel e é sempre observado nos meses de teste
    (jan–jun/2026 → jan–jun/2025), portanto não há preenchimento artificial.
    """
    chave = ["tipo", "periodo", "especialidade", "faixa"]
    teste = painel.loc[painel["particao"] == "teste", chave + ["estrato", "n", "lag_12"]]
    saz = teste.assign(modelo=SAZONAL, predito=teste["lag_12"].clip(lower=0).astype(float))
    saz = saz.drop(columns="lag_12")
    return pd.concat([prev, saz[prev.columns]], ignore_index=True)


def agrega(prev: pd.DataFrame) -> pd.DataFrame:
    """Reagrega célula → especialidade-mês (nível da decisão)."""
    a = (
        prev.groupby(["tipo", "modelo", "periodo", "especialidade"], observed=True)[["n", "predito"]]
        .sum()
        .reset_index()
    )
    a["erro_abs"] = (a["n"] - a["predito"]).abs()
    a["erro_quad"] = (a["n"] - a["predito"]) ** 2
    return a


def resumo(a: pd.DataFrame, por: list[str]) -> pd.DataFrame:
    r = a.groupby(por, observed=True).agg(
        MAE=("erro_abs", "mean"),
        RMSE=("erro_quad", lambda s: float(np.sqrt(s.mean()))),
        soma_erro=("erro_abs", "sum"),
        soma_obs=("n", "sum"),
        N=("n", "size"),
    ).reset_index()
    r["WAPE_%"] = 100 * r["soma_erro"] / r["soma_obs"]
    return r.drop(columns=["soma_erro", "soma_obs"])


def com_ganho(r: pd.DataFrame, por: list[str]) -> pd.DataFrame:
    """Ganho percentual de MAE sobre a persistência, dentro do mesmo recorte."""
    ref = r.loc[r["modelo"] == BASELINE, por + ["MAE"]].rename(columns={"MAE": "MAE_base"})
    r = r.merge(ref, on=por, how="left")
    r["ganho_%"] = 100 * (1 - r["MAE"] / r["MAE_base"])
    return r.drop(columns="MAE_base")


# --------------------------------------------------------------------------
# (d) comparação pareada
# --------------------------------------------------------------------------
def bootstrap_por_especialidade(d: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    """IC 95% da diferença média de |erro|, reamostrando ESPECIALIDADES.

    As 6 observações de uma especialidade são correlacionadas entre si; reamostrar
    especialidades inteiras (bootstrap por conglomerado) preserva essa dependência,
    o que um bootstrap linha a linha não faria e produziria intervalos estreitos
    demais.
    """
    por_esp = d.groupby("especialidade")["d"].agg(["sum", "size"])
    soma, tam = por_esp["sum"].to_numpy(), por_esp["size"].to_numpy()
    k = len(por_esp)
    idx = rng.integers(0, k, size=(BOOTSTRAP, k))
    medias = soma[idx].sum(axis=1) / tam[idx].sum(axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def comparacoes(a: pd.DataFrame) -> pd.DataFrame:
    pares = [
        ("Random Forest", "Gradient Boosting"),
        ("Random Forest", "Poisson"),
        ("Random Forest", BASELINE),
        ("Random Forest", SAZONAL),
        ("Gradient Boosting", BASELINE),
        ("Poisson", BASELINE),
        (SAZONAL, BASELINE),
    ]
    rng = np.random.default_rng(2026)
    linhas = []
    for tipo, g in a.groupby("tipo"):
        largo = g.pivot_table(index=["periodo", "especialidade"], columns="modelo", values="erro_abs")
        for m1, m2 in pares:
            d = (largo[m1] - largo[m2]).rename("d").reset_index()
            lo, hi = bootstrap_por_especialidade(d, rng)
            nz = d["d"][d["d"].abs() > 1e-12]
            p_wil = float(stats.wilcoxon(nz).pvalue) if len(nz) > 10 else np.nan
            linhas.append({
                "tipo": tipo, "modelo_A": m1, "modelo_B": m2,
                "dif_MAE_A_menos_B": d["d"].mean(),
                "IC95_inf": lo, "IC95_sup": hi,
                "IC_exclui_zero": (lo > 0) or (hi < 0),
                "A_melhor_em_%": 100 * (d["d"] < 0).mean(),
                "p_wilcoxon": p_wil,
            })
    return ajusta_holm(pd.DataFrame(linhas))


def ajusta_holm(comp: pd.DataFrame) -> pd.DataFrame:
    """Valores de p de Wilcoxon ajustados por Holm, dentro de cada base.

    São sete comparações por base; sem ajuste, a chance de alguma parecer
    significativa por acaso cresce com o número de testes.
    """
    comp = comp.copy()
    comp["p_holm"] = np.nan
    for _, g in comp.groupby("tipo"):
        ordem = g["p_wilcoxon"].sort_values()
        m, maior = len(ordem), 0.0
        for i, (idx, p) in enumerate(ordem.items()):
            maior = max(maior, min(1.0, (m - i) * p))
            comp.loc[idx, "p_holm"] = maior
    return comp


# --------------------------------------------------------------------------
# (a) premissas do Poisson
# --------------------------------------------------------------------------
def premissas_poisson(painel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    linhas_estrato, linhas_teste = [], []
    for tipo, p in painel.groupby("tipo"):
        treino = p[p["particao"] == "treino"]

        for estrato, g in list(treino.groupby("estrato", observed=True)) + [("todos", treino)]:
            media, var = g["n"].mean(), g["n"].var()
            linhas_estrato.append({
                "tipo": tipo, "estrato": estrato, "linhas": len(g),
                "media": media, "variancia": var,
                "var_sobre_media": var / media if media > 0 else np.nan,
                "zeros_%": 100 * (g["n"] == 0).mean(),
                "maximo": g["n"].max(),
            })

        # GLM ajustado exatamente como na 1ª origem (treino até dez/2025).
        X, y = matriz(treino)
        X = X.loc[:, X.std() > 0]
        glm = ajusta_poisson(X, y)
        mu = np.asarray(glm.fittedvalues, dtype=float)
        yv = y.to_numpy(float)

        # Cameron & Trivedi (1990): regressão auxiliar sem constante de
        # ((y − μ)² − y) / μ em μ. Coeficiente α > 0 e significativo indica
        # Var(y) = μ + α·μ², isto é, sobredispersão.
        z = ((yv - mu) ** 2 - yv) / mu
        aux = sm.OLS(z, mu).fit()

        zeros_obs = (yv == 0).mean()
        zeros_esp = np.exp(-mu).mean()  # P(Y=0) = e^(−μ) sob Poisson

        estratos = treino["estrato"].astype(str).to_numpy()
        por_estrato = {
            e: (100 * (yv[estratos == e] == 0).mean(), 100 * np.exp(-mu[estratos == e]).mean())
            for e in ("baixo", "medio", "alto")
        }

        linhas_teste.append({
            "tipo": tipo,
            "linhas_treino": len(yv),
            "pearson_chi2": glm.pearson_chi2,
            "gl_residuo": glm.df_resid,
            "pearson_chi2_sobre_gl": glm.pearson_chi2 / glm.df_resid,
            "deviance_sobre_gl": glm.deviance / glm.df_resid,
            "cameron_trivedi_alfa": float(aux.params[0]),
            "cameron_trivedi_t": float(aux.tvalues[0]),
            "cameron_trivedi_p": float(aux.pvalues[0]),
            "zeros_obs_%": 100 * zeros_obs,
            "zeros_esperados_poisson_%": 100 * zeros_esp,
            **{f"zeros_obs_{e}_%": v[0] for e, v in por_estrato.items()},
            **{f"zeros_esp_{e}_%": v[1] for e, v in por_estrato.items()},
        })
    return pd.DataFrame(linhas_estrato), pd.DataFrame(linhas_teste)


# --------------------------------------------------------------------------
# (e) importância por permutação, versão detalhada
# --------------------------------------------------------------------------
HISTORICO = "historico (lags + medias)"


def importancia_detalhada(painel: pd.DataFrame) -> pd.DataFrame:
    """Reajusta os mesmos modelos de `modelagem.py` (mesmas sementes) e mede a
    importância com mais repetições, guardando cada repetição.

    Diferenças em relação à Tabela 12 original, que usava 5 repetições e
    truncava negativos em zero antes de normalizar:
      - 10 repetições × 6 origens = 60 medições por variável e modelo;
      - o valor bruto (aumento do MAE, em atendimentos) é guardado sem truncar;
      - além das variáveis individuais, o BLOCO de histórico (lag_1, lag_2,
        lag_3, lag_12, media_3m, media_6m) é permutado em conjunto. Quando
        variáveis são substituíveis, permutar uma de cada vez subestima a
        relevância do conjunto, porque o modelo recorre às outras.
    """
    linhas = []
    for tipo, p in painel.groupby("tipo"):
        origens = sorted(p.loc[p["particao"] == "teste", "periodo"].unique())
        for i, origem in enumerate(origens):
            treino, teste = p[p["periodo"] < origem], p[p["periodo"] == origem]
            Xtr, ytr = matriz(treino)
            Xte, yte = matriz(teste)
            Xtr = Xtr.loc[:, Xtr.std() > 0]
            Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)

            modelos = {
                "Poisson": ajusta_poisson(Xtr, ytr),
                "Random Forest": RandomForestRegressor(
                    n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
                ).fit(Xtr, ytr),
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
                ).fit(Xtr, ytr),
            }

            grupos: dict[str, list[str]] = {}
            for c in Xte.columns:
                grupos.setdefault(origem_da_coluna(c), []).append(c)
            grupos[HISTORICO] = [c for c in Xte.columns if c.startswith(("lag_", "media_"))]

            for nome, modelo in modelos.items():
                rng = np.random.default_rng(1000 + i)
                base = mean_absolute_error(yte, preve(nome, modelo, Xte))
                for variavel, colunas in grupos.items():
                    for r in range(REPETICOES_PERMUTACAO):
                        Xp = Xte.copy()
                        Xp[colunas] = Xp[colunas].to_numpy()[rng.permutation(len(Xp))]
                        perda = mean_absolute_error(yte, preve(nome, modelo, Xp)) - base
                        linhas.append({"tipo": tipo, "modelo": nome, "origem": origem,
                                       "repeticao": r, "variavel": variavel, "aumento_MAE": perda})
            print(f"  importância {tipo} origem {pd.Timestamp(origem):%m/%Y} ok", flush=True)

    bruto = pd.DataFrame(linhas)
    r = bruto.groupby(["tipo", "modelo", "variavel"])["aumento_MAE"].agg(
        media="mean", desvio="std", p05=lambda s: s.quantile(0.05), p95=lambda s: s.quantile(0.95),
        negativas_pct=lambda s: 100 * (s < 0).mean(),
    ).reset_index()

    # Normalização explícita: peso = max(média, 0) / Σ max(média, 0), somando
    # apenas as variáveis INDIVIDUAIS (o bloco de histórico não entra no
    # denominador, porque se sobrepõe a elas).
    indiv = r["variavel"] != HISTORICO
    pos = r["media"].clip(lower=0).where(indiv)
    r["peso_%"] = 100 * pos / pos.groupby([r["tipo"], r["modelo"]]).transform("sum")
    return r


# --------------------------------------------------------------------------
def main() -> int:
    import sys
    rapido = "--rapido" in sys.argv  # pula a importância, que reajusta 36 modelos
    pd.set_option("display.width", 200)
    fmt = "{:.4f}".format
    painel = pd.read_csv(PROCESSADO / "painel.csv.gz", parse_dates=["periodo"])
    prev = pd.read_csv(TABELAS / "previsoes.csv", parse_dates=["periodo"])
    prev = acrescenta_sazonal(prev, painel)
    a = agrega(prev)

    # (f)+(d) agregado com 4 casas, incluindo o sazonal
    geral = com_ganho(resumo(a, ["tipo", "modelo"]), ["tipo"])
    # (c) por origem
    mensal = com_ganho(resumo(a, ["tipo", "periodo", "modelo"]), ["tipo", "periodo"])
    # (d) erro relativo por especialidade
    esp = resumo(a, ["tipo", "especialidade", "modelo"])
    vol = a[a["modelo"] == BASELINE].groupby(["tipo", "especialidade"])["n"].mean().rename("media_mensal")
    esp = com_ganho(esp, ["tipo", "especialidade"]).merge(vol.reset_index(), on=["tipo", "especialidade"])
    # (f) estrato com 4 casas (célula), incluindo o sazonal
    prev["erro_abs"] = (prev["n"] - prev["predito"]).abs()
    estrato = prev.groupby(["tipo", "estrato", "modelo"]).agg(
        MAE=("erro_abs", "mean"), N=("n", "size"), media_obs=("n", "mean")
    ).reset_index()

    # (d) relevância operacional: WAPE do RF por terço de volume das especialidades
    rf = esp[esp["modelo"] == "Random Forest"].copy()
    rf["terco_volume"] = rf.groupby("tipo")["media_mensal"].transform(
        lambda s: pd.qcut(s, 3, labels=["menor volume", "intermediário", "maior volume"])
    )
    tercos = rf.groupby(["tipo", "terco_volume"], observed=True).agg(
        especialidades=("especialidade", "size"),
        volume_mediano=("media_mensal", "median"),
        WAPE_mediano=("WAPE_%", "median"),
        MAE_mediano=("MAE", "median"),
    ).reset_index()
    tercos.to_csv(TABELAS / "validacao_wape_tercos.csv", index=False)
    print("\n=== WAPE DO RANDOM FOREST POR TERÇO DE VOLUME ===\n", tercos.to_string(index=False, float_format=fmt))

    comp = comparacoes(a)
    premissas_estrato, premissas_teste = premissas_poisson(painel)

    geral.to_csv(TABELAS / "validacao_geral.csv", index=False)
    mensal.to_csv(TABELAS / "validacao_por_origem.csv", index=False)
    esp.to_csv(TABELAS / "validacao_por_especialidade.csv", index=False)
    estrato.to_csv(TABELAS / "validacao_estrato.csv", index=False)
    comp.to_csv(TABELAS / "validacao_comparacao_pareada.csv", index=False)
    premissas_estrato.to_csv(TABELAS / "validacao_poisson_estratos.csv", index=False)
    premissas_teste.to_csv(TABELAS / "validacao_poisson_testes.csv", index=False)

    print("\n=== GERAL (especialidade-mês) ===\n", geral.to_string(index=False, float_format=fmt))
    print("\n=== POR ORIGEM ===\n", mensal.to_string(index=False, float_format=fmt))
    print("\n=== ESTRATO (célula) ===\n", estrato.to_string(index=False, float_format=fmt))
    print("\n=== COMPARAÇÃO PAREADA ===\n", comp.to_string(index=False, float_format=fmt))
    print("\n=== POISSON: ESTRATOS (treino) ===\n", premissas_estrato.to_string(index=False, float_format=fmt))
    print("\n=== POISSON: TESTES ===\n", premissas_teste.T.to_string())

    if rapido:
        return 0
    print("\ncalculando importância detalhada (reajusta os modelos)...", flush=True)
    imp = importancia_detalhada(painel)
    imp.to_csv(TABELAS / "validacao_importancia.csv", index=False)
    print("\n=== IMPORTÂNCIA ===\n", imp.sort_values(["tipo", "modelo", "media"], ascending=[True, True, False])
          .to_string(index=False, float_format=fmt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
