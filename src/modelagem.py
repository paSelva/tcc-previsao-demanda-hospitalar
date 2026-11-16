"""
Modelagem e avaliação (CRISP-DM — Fases 4 e 5).

Treina e compara quatro modelos, separadamente para consultas e internações:

    1. Baseline  — persistência: a previsão do mês é o valor do mês anterior
                   no mesmo grupo. É a referência honesta para previsão de
                   curto prazo; média global não serve.
    2. Poisson   — GLM família Poisson, ligação log (statsmodels).
    3. Random Forest Regressor.
    4. Gradient Boosting Regressor.

Avaliação por origem rolante
----------------------------
Com apenas 30 meses modeláveis, um corte único treino/teste é frágil: o
resultado dependeria de qual mês caiu no teste. Usa-se janela expansiva —
para cada um dos 6 meses de teste, treina-se com tudo que veio antes e
prevê-se aquele mês. São 6 origens por modelo e por base.

Duas providências exigidas pela exploratória
--------------------------------------------
As médias de célula vão de 0,0 a 126,7. Uma métrica única seria dominada
pelo estrato mais numeroso e não diria qual modelo é melhor. Portanto:

    (a) as previsões são REAGREGADAS para especialidade-mês antes de pontuar,
        que é o nível da decisão apoiada ("quantas vagas abrir por
        especialidade");
    (b) as métricas também são reportadas ESTRATIFICADAS por volume, com os
        estratos definidos apenas com dados de treino.

Uso:
    ./venv/bin/python src/modelagem.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent.parent
PROCESSADO = RAIZ / "data" / "processed"
TABELAS = RAIZ / "outputs" / "tabelas"

# Colunas que identificam a linha ou o resultado, e não entram como preditor.
# `especialidade` e `faixa` são chave do painel mas TAMBÉM são preditores, por
# isso ficam de fora desta lista: viram dummies em `matriz`.
NAO_PREDITORES = ["tipo", "periodo", "n", "particao", "estrato"]
CATEGORICAS = ["especialidade", "faixa"]

# Defasagens e médias móveis entram em log(1+x) para todos os modelos. Para o
# Poisson isso é o que torna o efeito multiplicativo, coerente com a ligação
# log; para as árvores é indiferente, porque são invariantes a transformação
# monótona. Usar a mesma matriz nos quatro modelos mantém a comparação justa.
def transforma(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if c.startswith(("lag_", "media_")):
            X[c] = np.log1p(X[c].clip(lower=0))
    return X


def matriz(painel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = painel["n"]
    X = painel.drop(columns=[c for c in NAO_PREDITORES if c in painel.columns])
    # drop_first=True é obrigatório: com todos os níveis mais a constante do
    # GLM a matriz fica singular e o IRLS do Poisson não converge.
    X = pd.get_dummies(X, columns=CATEGORICAS, drop_first=True, dtype=float)
    return transforma(X), y


def origem_da_coluna(coluna: str) -> str:
    """Mapeia coluna one-hot de volta à variável de origem, para a PI2."""
    for cat in CATEGORICAS:
        if coluna.startswith(f"{cat}_"):
            return cat
    return coluna


def ajusta_poisson(X: pd.DataFrame, y: pd.Series) -> sm.GLM:
    Xc = sm.add_constant(X, has_constant="add")
    modelo = sm.GLM(y, Xc, family=sm.families.Poisson())
    return modelo.fit(maxiter=200)


def preve(nome: str, modelo, X: pd.DataFrame) -> np.ndarray:
    if nome == "Poisson":
        p = modelo.predict(sm.add_constant(X, has_constant="add"))
    else:
        p = modelo.predict(X)
    return np.clip(np.asarray(p, dtype=float), 0, None)


def importancia_permutacao(
    nome: str, modelo, X: pd.DataFrame, y: pd.Series, repeticoes: int = 5
) -> pd.Series:
    """Importância por permutação, agrupada por variável de origem.

    Medida escolhida por ser **model-agnostic**: mede o quanto o erro piora ao
    embaralhar uma variável, o que é comparável entre Poisson, Random Forest e
    Gradient Boosting. O caminho alternativo — |coeficiente| no GLM e
    `feature_importances_` nas árvores — não é comparável entre modelos nem
    entre variáveis de escalas diferentes, e faria `especialidade` dominar
    apenas por estar espalhada em 69 colunas dummy.

    Colunas one-hot da mesma variável são permutadas EM BLOCO, preservando a
    linha: embaralhar cada dummy isoladamente produziria linhas impossíveis
    (duas especialidades ao mesmo tempo, ou nenhuma).
    """
    rng = np.random.default_rng(42)
    erro_base = mean_absolute_error(y, preve(nome, modelo, X))

    grupos: dict[str, list[str]] = {}
    for coluna in X.columns:
        grupos.setdefault(origem_da_coluna(coluna), []).append(coluna)

    resultado = {}
    for variavel, colunas in grupos.items():
        perdas = []
        for _ in range(repeticoes):
            Xp = X.copy()
            ordem = rng.permutation(len(Xp))
            Xp[colunas] = Xp[colunas].to_numpy()[ordem]
            perdas.append(mean_absolute_error(y, preve(nome, modelo, Xp)) - erro_base)
        resultado[variavel] = max(float(np.mean(perdas)), 0.0)
    return pd.Series(resultado)


def metricas(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        # A deviance de Poisson exige previsão estritamente positiva.
        "Deviance": mean_poisson_deviance(y, np.clip(p, 1e-9, None)),
    }


def roda(painel: pd.DataFrame, tipo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    painel = painel.loc[painel["tipo"] == tipo].copy()
    periodos = sorted(painel["periodo"].unique())
    origens = sorted(painel.loc[painel["particao"] == "teste", "periodo"].unique())

    previsoes, importancias = [], []

    for origem in origens:
        treino = painel[painel["periodo"] < origem]
        teste = painel[painel["periodo"] == origem]

        Xtr, ytr = matriz(treino)
        Xte, _ = matriz(teste)
        # Colunas constantes na janela de treino não carregam informação e
        # tornam o design singular — saem antes do ajuste.
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

        base = teste.assign(
            modelo="Baseline", predito=teste["lag_1"].clip(lower=0).to_numpy(float)
        )
        previsoes.append(base[["periodo", "especialidade", "faixa", "estrato", "n", "modelo", "predito"]])

        for nome, modelo in modelos.items():
            p = preve(nome, modelo, Xte)
            previsoes.append(
                teste.assign(modelo=nome, predito=p)[
                    ["periodo", "especialidade", "faixa", "estrato", "n", "modelo", "predito"]
                ]
            )

            imp = importancia_permutacao(nome, modelo, Xte, teste["n"])
            importancias.append(
                imp.rename("valor").reset_index().assign(modelo=nome, origem=origem)
            )

    previsoes = pd.concat(previsoes, ignore_index=True)
    importancias = pd.concat(importancias, ignore_index=True)

    # Importância média entre as origens, normalizada para somar 1 por modelo.
    imp = importancias.groupby(["modelo", "index"])["valor"].mean().reset_index()
    imp["peso"] = imp.groupby("modelo")["valor"].transform(lambda s: s / s.sum())
    imp = imp.rename(columns={"index": "variavel"}).assign(tipo=tipo)

    previsoes["tipo"] = tipo
    return previsoes, imp


def avalia(prev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    linhas_celula, linhas_agreg, linhas_estrato = [], [], []

    for (tipo, modelo), g in prev.groupby(["tipo", "modelo"]):
        linhas_celula.append({"tipo": tipo, "modelo": modelo, **metricas(g["n"].to_numpy(float), g["predito"].to_numpy())})

        # (a) reagregação para especialidade-mês: o nível da decisão apoiada.
        a = g.groupby(["periodo", "especialidade"], observed=True)[["n", "predito"]].sum()
        linhas_agreg.append({"tipo": tipo, "modelo": modelo, **metricas(a["n"].to_numpy(float), a["predito"].to_numpy())})

        # (b) estratificação por volume.
        for estrato, ge in g.groupby("estrato", observed=True):
            linhas_estrato.append(
                {"tipo": tipo, "modelo": modelo, "estrato": estrato,
                 "media_obs": ge["n"].mean(),
                 **metricas(ge["n"].to_numpy(float), ge["predito"].to_numpy())}
            )

    return (pd.DataFrame(linhas_celula), pd.DataFrame(linhas_agreg), pd.DataFrame(linhas_estrato))


def mostra(titulo: str, df: pd.DataFrame, chave: list[str]) -> None:
    print(f"\n{titulo}")
    for tipo, g in df.groupby("tipo"):
        print(f"\n  {tipo.upper()}")
        g = g.sort_values("MAE")
        print("    " + g[chave + ["MAE", "RMSE", "Deviance"]].to_string(index=False, float_format="%.3f").replace("\n", "\n    "))


def main() -> int:
    painel = pd.read_csv(PROCESSADO / "painel.csv.gz", parse_dates=["periodo"])
    TABELAS.mkdir(parents=True, exist_ok=True)

    todas_prev, todas_imp = [], []
    for tipo in ("consulta", "internacao"):
        print(f"treinando {tipo}...", flush=True)
        p, i = roda(painel, tipo)
        todas_prev.append(p)
        todas_imp.append(i)

    prev = pd.concat(todas_prev, ignore_index=True)
    imp = pd.concat(todas_imp, ignore_index=True)
    celula, agreg, estrato = avalia(prev)

    prev.to_csv(TABELAS / "previsoes.csv", index=False)
    celula.to_csv(TABELAS / "metricas_celula.csv", index=False)
    agreg.to_csv(TABELAS / "metricas_agregado.csv", index=False)
    estrato.to_csv(TABELAS / "metricas_estrato.csv", index=False)
    imp.to_csv(TABELAS / "importancia_variaveis.csv", index=False)

    mostra("=== NÍVEL DA CÉLULA (mês × especialidade × faixa) ===", celula, ["modelo"])
    mostra("=== REAGREGADO PARA ESPECIALIDADE-MÊS (nível da decisão) ===", agreg, ["modelo"])
    mostra("=== POR ESTRATO DE VOLUME ===", estrato.sort_values(["estrato", "MAE"]), ["estrato", "modelo", "media_obs"])

    print("\n=== RELEVÂNCIA PREDITIVA — top 8 por modelo (peso %) ===")
    print("Leitura: contribuição para a capacidade preditiva. NÃO é relação de causa e efeito.")
    for (tipo, modelo), g in imp.groupby(["tipo", "modelo"]):
        top = g.nlargest(8, "peso")
        linha = "  ".join(f"{r.variavel}={100*r.peso:.1f}" for r in top.itertuples())
        print(f"\n  {tipo[:5]} · {modelo}\n    {linha}")

    print(f"\nTabelas gravadas em {TABELAS.relative_to(RAIZ)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
