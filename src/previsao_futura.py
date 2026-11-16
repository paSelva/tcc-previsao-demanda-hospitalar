"""
Previsão um mês à frente (CRISP-DM — insumo da Fase 6).

`modelagem.py` produz o *backtest*: previsões para os 6 meses de teste, que
já aconteceram, e cujo erro pôde ser medido. O painel também precisa da
previsão do **próximo** mês, que ainda não ocorreu.

Horizonte de um mês, e só
-------------------------
Todas as defasagens do mês seguinte são conhecidas: `lag_1` é o último mês
observado, `media_3m` são os três últimos, e assim por diante. Nenhuma
previsão precisa ser realimentada como se fosse observação.

Horizontes de 3 ou 6 meses exigiriam previsão recursiva — usar a previsão de
julho como se fosse o valor observado para prever agosto. O erro se acumula de
forma não medida por este trabalho, que avaliou apenas um passo à frente.
Por isso o painel oferece **um mês**, e não 3 ou 6 (seção 4.5.5 do TCC).

Uso:
    ./venv/bin/python src/previsao_futura.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from modelagem import ajusta_poisson, matriz, preve
from painel import DEFASAGENS, JANELAS_MOVEIS

RAIZ = Path(__file__).resolve().parent.parent
PROCESSADO = RAIZ / "data" / "processed"
TABELAS = RAIZ / "outputs" / "tabelas"


def dias_uteis(periodo: pd.Timestamp) -> int:
    fim = periodo + pd.offsets.MonthEnd(0)
    return int(np.busday_count(periodo.date(), (fim + pd.Timedelta(days=1)).date()))


def composicao_do_mes(atend: pd.DataFrame, tipo: str, mes: pd.Timestamp) -> pd.DataFrame:
    """Composição demográfica observada no último mês, que vira o t-1 da previsão."""
    sub = atend[(atend["tipo"] == tipo) & (atend["periodo"] == mes)]
    g = sub.groupby(["especialidade", "faixa"], observed=True)
    return pd.DataFrame(
        {
            "idade_media_lag1": g["idade"].mean(),
            "prop_feminino_lag1": g["sexo"].apply(lambda s: (s == "FEMININO").mean()),
            "prop_natal_lag1": g["municipio"].apply(lambda s: (s == "NATAL").mean()),
            "prop_mun_ni_lag1": g["municipio"].apply(lambda s: (s == "NAO INFORMADO").mean()),
            "n_municipios_lag1": g["municipio"].nunique(),
        }
    ).reset_index()


def linha_do_proximo_mes(painel: pd.DataFrame, atend: pd.DataFrame, tipo: str) -> pd.DataFrame:
    """Monta a matriz de preditores do mês seguinte ao último observado."""
    p = painel[painel["tipo"] == tipo]
    ultimo = p["periodo"].max()
    alvo = ultimo + pd.DateOffset(months=1)

    # Série de contagens por grupo, para gerar as defasagens sem ambiguidade.
    serie = p.pivot_table(
        index="periodo", columns=["especialidade", "faixa"], values="n", observed=True
    ).sort_index()

    linhas = {f"lag_{d}": serie.iloc[-d] for d in DEFASAGENS}
    for j in JANELAS_MOVEIS:
        linhas[f"media_{j}m"] = serie.iloc[-j:].mean()

    novo = pd.DataFrame(linhas)
    novo.index.names = ["especialidade", "faixa"]
    novo = novo.reset_index()

    comp = composicao_do_mes(atend, tipo, ultimo)
    novo = novo.merge(comp, on=["especialidade", "faixa"], how="left")

    # Grupo sem atendimento no último mês não tem composição observável;
    # mesma regra da construção do painel: média do histórico do próprio grupo.
    for coluna in [c for c in novo.columns if c.endswith("_lag1")]:
        novo[coluna] = novo[coluna].fillna(p[coluna].mean())

    novo["mes"] = alvo.month
    novo["ano"] = alvo.year
    novo["trimestre"] = alvo.quarter
    novo["dias_uteis"] = dias_uteis(alvo)
    novo["periodo"] = alvo
    novo["tipo"] = tipo
    novo["n"] = 0.0  # marcador; não é usado, o alvo é desconhecido

    estratos = p.groupby(["especialidade", "faixa"], observed=True)["estrato"].last()
    novo = novo.merge(estratos, on=["especialidade", "faixa"], how="left")
    return novo, ultimo, alvo


def main() -> int:
    painel = pd.read_csv(PROCESSADO / "painel.csv.gz", parse_dates=["periodo"])
    atend = pd.read_csv(PROCESSADO / "atendimentos.csv.gz", parse_dates=["data_hora"])
    atend["periodo"] = atend["data_hora"].dt.to_period("M").dt.to_timestamp()
    atend = atend.loc[atend["idade"].notna()].copy()
    atend["faixa"] = pd.cut(
        atend["idade"], bins=[-1, 14, 29, 44, 59, 74, 200],
        labels=["0-14", "15-29", "30-44", "45-59", "60-74", "75+"],
    )

    saida = []
    for tipo in ("consulta", "internacao"):
        treino = painel[painel["tipo"] == tipo]
        novo, ultimo, alvo = linha_do_proximo_mes(painel, atend, tipo)

        Xtr, ytr = matriz(treino)
        Xtr = Xtr.loc[:, Xtr.std() > 0]
        Xno, _ = matriz(novo)
        Xno = Xno.reindex(columns=Xtr.columns, fill_value=0.0)

        modelos = {
            "Poisson": ajusta_poisson(Xtr, ytr),
            "Random Forest": RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
            ).fit(Xtr, ytr),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
            ).fit(Xtr, ytr),
        }

        base = novo.assign(modelo="Baseline", predito=novo["lag_1"].clip(lower=0))
        saida.append(base)
        for nome, modelo in modelos.items():
            saida.append(novo.assign(modelo=nome, predito=preve(nome, modelo, Xno)))

        print(f"{tipo}: último mês observado {ultimo:%m/%Y} → previsão para {alvo:%m/%Y}")

    colunas = ["tipo", "periodo", "especialidade", "faixa", "estrato", "modelo", "predito"]
    prev = pd.concat(saida, ignore_index=True)[colunas]
    prev["predito"] = prev["predito"].round(1)

    TABELAS.mkdir(parents=True, exist_ok=True)
    prev.to_csv(TABELAS / "previsao_futura.csv", index=False)

    print(f"\nGravado: outputs/tabelas/previsao_futura.csv ({len(prev):,} linhas)".replace(",", "."))
    resumo = prev.groupby(["tipo", "modelo"])["predito"].sum().round(0)
    print("\nTotal previsto por modelo:")
    print(resumo.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
