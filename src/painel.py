"""
Construção do painel de contagens (CRISP-DM — Preparação dos Dados).

Transforma os microdados padronizados em `data/processed/atendimentos.csv.gz`
no painel que alimenta os modelos, com a mesma estrutura para as duas bases —
condição para que a comparação da PI1 seja legítima.

Unidade de análise
------------------
    1 linha = mês × especialidade × faixa etária
    alvo    = número de atendimentos naquele grupo e mês

A escolha dessa chave é justificada pela medição de esparsidade em
`docs/02-unidade-de-analise.md`: o painel de cinco dimensões chega a 98,2% de
células vazias nas internações, regime em que o excesso de zeros deixa de ser
característica do fenômeno e vira artefato do recorte.

`sexo` e `município` não entram na chave, mas não são descartados: retornam
como **composição defasada** do grupo (proporção feminina, proporção de Natal,
idade média etc.), medida sempre em t-1 ou antes.

Vazamento temporal
------------------
Toda variável derivada é defasada. Nada que só se saiba depois de o mês
terminar entra na previsão desse mês. A divisão treino/teste é por período,
nunca por linha.

Uso:
    ./venv/bin/python src/painel.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PROCESSADO = RAIZ / "data" / "processed"

FAIXAS_LIMITES = [-1, 14, 29, 44, 59, 74, 200]
FAIXAS_ROTULOS = ["0-14", "15-29", "30-44", "45-59", "60-74", "75+"]

# Defasagens usadas como preditores. A de 12 meses capta sazonalidade anual e
# custa o primeiro ano de histórico, que passa a servir só para gerá-la.
DEFASAGENS = [1, 2, 3, 12]
JANELAS_MOVEIS = [3, 6]

# Últimos meses reservados para teste (ordem temporal preservada).
MESES_TESTE = 6


def carrega() -> pd.DataFrame:
    df = pd.read_csv(PROCESSADO / "atendimentos.csv.gz", parse_dates=["data_hora"])

    # Registros sem idade válida não podem receber faixa etária, que é dimensão
    # da chave. São excluídos aqui e contabilizados no relatório da ingestão —
    # mantê-los criaria um nível "não informado" quase vazio em toda a grade.
    sem_idade = df["idade"].isna()
    print(f"  registros sem idade válida, excluídos do painel: {int(sem_idade.sum())}")
    df = df.loc[~sem_idade].copy()

    df["periodo"] = df["data_hora"].dt.to_period("M").dt.to_timestamp()
    df["faixa"] = pd.cut(df["idade"], bins=FAIXAS_LIMITES, labels=FAIXAS_ROTULOS)
    return df


def grade_completa(sub: pd.DataFrame) -> pd.DataFrame:
    """Produto cartesiano período × especialidade × faixa, com zeros explícitos.

    Um mês sem nenhuma internação de pediatria em 15-29 anos é informação —
    contagem zero —, não ausência de dado. Sem a grade completa o modelo só
    veria os grupos que ocorreram, o que enviesaria a previsão para cima.
    """
    eixos = pd.MultiIndex.from_product(
        [
            sorted(sub["periodo"].unique()),
            sorted(sub["especialidade"].unique()),
            FAIXAS_ROTULOS,
        ],
        names=["periodo", "especialidade", "faixa"],
    )
    contagem = sub.groupby(["periodo", "especialidade", "faixa"], observed=True).size()
    return contagem.reindex(eixos, fill_value=0).rename("n").reset_index()


def composicao(sub: pd.DataFrame) -> pd.DataFrame:
    """Características demográficas do grupo em cada mês, antes de defasar."""
    g = sub.groupby(["periodo", "especialidade", "faixa"], observed=True)
    return pd.DataFrame(
        {
            "idade_media": g["idade"].mean(),
            "prop_feminino": g["sexo"].apply(lambda s: (s == "FEMININO").mean()),
            "prop_natal": g["municipio"].apply(lambda s: (s == "NATAL").mean()),
            "prop_mun_ni": g["municipio"].apply(lambda s: (s == "NAO INFORMADO").mean()),
            "n_municipios": g["municipio"].nunique(),
        }
    ).reset_index()


def dias_uteis(periodo: pd.Timestamp) -> int:
    fim = periodo + pd.offsets.MonthEnd(0)
    return int(np.busday_count(periodo.date(), (fim + pd.Timedelta(days=1)).date()))


def monta(sub: pd.DataFrame, tipo: str) -> pd.DataFrame:
    painel = grade_completa(sub).merge(
        composicao(sub), on=["periodo", "especialidade", "faixa"], how="left"
    )
    painel = painel.sort_values(["especialidade", "faixa", "periodo"])
    grupo = painel.groupby(["especialidade", "faixa"], observed=True)

    # --- defasagens do próprio alvo ---
    for d in DEFASAGENS:
        painel[f"lag_{d}"] = grupo["n"].shift(d)
    for j in JANELAS_MOVEIS:
        painel[f"media_{j}m"] = grupo["n"].transform(
            lambda s, j=j: s.shift(1).rolling(j).mean()
        )

    # --- composição, sempre defasada em um mês ---
    for coluna in ("idade_media", "prop_feminino", "prop_natal", "prop_mun_ni", "n_municipios"):
        painel[f"{coluna}_lag1"] = grupo[coluna].shift(1)
        painel = painel.drop(columns=coluna)

    # --- calendário (conhecido de antemão, não precisa defasar) ---
    painel["mes"] = painel["periodo"].dt.month
    painel["ano"] = painel["periodo"].dt.year
    painel["trimestre"] = painel["periodo"].dt.quarter
    painel["dias_uteis"] = painel["periodo"].map(dias_uteis)

    # O primeiro ano vira insumo das defasagens e sai da amostra de modelagem.
    corte = painel["periodo"].min() + pd.DateOffset(months=max(DEFASAGENS))
    painel = painel.loc[painel["periodo"] >= corte].copy()

    # Composição defasada é NaN onde o grupo não teve atendimento em t-1.
    # Zero atendimentos significa "sem composição observável": preenchemos com
    # a média do grupo no treino, decisão registrada em 4.2.
    periodos = sorted(painel["periodo"].unique())
    inicio_teste = periodos[-MESES_TESTE]
    painel["particao"] = np.where(painel["periodo"] < inicio_teste, "treino", "teste")

    treino = painel["particao"] == "treino"
    for coluna in [c for c in painel.columns if c.endswith("_lag1")]:
        painel[coluna] = painel[coluna].fillna(painel.loc[treino, coluna].mean())
    for coluna in [c for c in painel.columns if c.startswith(("lag_", "media_"))]:
        painel[coluna] = painel[coluna].fillna(0)

    # Estrato de volume, definido SÓ com o treino para não vazar o teste.
    # Serve para reportar erro por faixa de volume (seção 4.4.3) em vez de uma
    # métrica única dominada pelos grupos mais numerosos.
    media_treino = painel.loc[treino].groupby(["especialidade", "faixa"], observed=True)["n"].mean()
    cortes = media_treino.quantile([1 / 3, 2 / 3]).values
    estrato = pd.cut(
        media_treino, [-1, *cortes, np.inf], labels=["baixo", "medio", "alto"]
    ).rename("estrato")
    painel = painel.merge(estrato, on=["especialidade", "faixa"], how="left")

    painel.insert(0, "tipo", tipo)
    return painel.reset_index(drop=True)


def main() -> int:
    df = carrega()
    paineis = []

    for tipo in ("consulta", "internacao"):
        painel = monta(df.loc[df["tipo"] == tipo].copy(), tipo)
        paineis.append(painel)

        periodos = painel["periodo"].nunique()
        zeros = 100 * (painel["n"] == 0).mean()
        teste = painel["particao"] == "teste"
        print(
            f"\n{tipo.upper()}\n"
            f"  linhas          {len(painel):>8,}".replace(",", ".") + "\n"
            f"  meses           {periodos:>8}  "
            f"({painel['periodo'].min():%m/%Y} a {painel['periodo'].max():%m/%Y})\n"
            f"  especialidades  {painel['especialidade'].nunique():>8}\n"
            f"  zeros no alvo   {zeros:>7.1f}%\n"
            f"  alvo            média {painel['n'].mean():.1f}  "
            f"mediana {painel['n'].median():.0f}  máx {painel['n'].max():.0f}\n"
            f"  treino/teste    {(~teste).sum():,} / {teste.sum():,}".replace(",", ".")
        )

    painel = pd.concat(paineis, ignore_index=True)
    destino = PROCESSADO / "painel.csv.gz"
    painel.to_csv(destino, index=False, compression="gzip")
    print(f"\nGravado: {destino.relative_to(RAIZ)}")
    print("Preditores:", ", ".join(
        c for c in painel.columns
        if c not in {"tipo", "periodo", "especialidade", "faixa", "n", "particao", "estrato"}
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
