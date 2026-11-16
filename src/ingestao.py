"""
Ingestão e padronização das bases do HUOL (CRISP-DM — Preparação dos Dados).

Lê os CSVs brutos de `data/raw/`, resolve os problemas de qualidade levantados em
`docs/01-entendimento-dos-dados.md` e grava um arquivo único e padronizado em
`data/processed/`.

Cada tratamento aplicado é contabilizado e impresso ao final: esse relatório é a
evidência que sustenta a seção 4.2 do TCC. Nada é corrigido silenciosamente.

Uso:
    ./venv/bin/python src/ingestao.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
BRUTO = RAIZ / "data" / "raw"
PROCESSADO = RAIZ / "data" / "processed"

ENCODING_FONTE = "ISO-8859-1"
COLUNAS_PADRAO = ["data_hora", "especialidade", "municipio", "idade", "sexo"]

# Idades acima deste limite são tratadas como implausíveis (ver nota 3.8).
# A base de consultas registra um caso de 124 anos.
IDADE_MAXIMA_PLAUSIVEL = 115

# Nota 3.2: a codificação de sexo muda entre o arquivo anual de 2023 e os
# trimestrais. Mapeamento explícito para um vocabulário único.
MAPA_SEXO = {
    "F": "FEMININO",
    "FEMININO": "FEMININO",
    "M": "MASCULINO",
    "MASCULINO": "MASCULINO",
    "I": "INDEFINIDO",
    "INDEFINIDO": "INDEFINIDO",
    "INFEFINIDO": "INDEFINIDO",  # erro de digitação na fonte
}

NAO_INFORMADO = "NAO INFORMADO"

ocorrencias: Counter = Counter()


def normaliza_texto(valor: object) -> str:
    """Remove acentos, colapsa espaços e padroniza em caixa alta.

    Nota 3.9: o mesmo município aparece como 'NATAL' e 'Macaíba'. Sem isso o
    painel duplicaria linhas para a mesma categoria.
    """
    if pd.isna(valor):
        return ""
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto).strip().upper()


def le_arquivo(caminho: Path) -> pd.DataFrame:
    """Lê um CSV bruto, tratando os três layouts encontrados na fonte."""
    with caminho.open(encoding=ENCODING_FONTE) as fh:
        primeira = fh.readline()

    # Nota 3.5: os arquivos anuais de 2023 trazem 4 linhas de metadados antes
    # do cabeçalho, que fica na linha 5 e usa outra grafia.
    linhas_preambulo = 4 if primeira.startswith("NOME DA BASE") else 0

    df = pd.read_csv(
        caminho,
        sep=";",
        encoding=ENCODING_FONTE,
        skiprows=linhas_preambulo,
        dtype=str,
        keep_default_na=False,
    )
    df.columns = [normaliza_texto(c) for c in df.columns]

    # Nota 3.4: um único arquivo de consultas traz 8 colunas, com 'sequencial',
    # 'datanasci' e 'Zona' a mais. São descartadas: variável presente em um só
    # trimestre não pode entrar no modelo.
    if "SEQUENCIAL" in df.columns:
        ocorrencias["registros_com_colunas_extras"] += len(df)

    coluna_data = next(c for c in df.columns if "DATA" in c and "NASCI" not in c)
    colunas = [coluna_data, "ESPECIALIDADE", "MUNICIPIO", "IDADE", "SEXO"]

    faltantes = [c for c in colunas if c not in df.columns]
    if faltantes:
        raise ValueError(f"{caminho.name}: colunas ausentes {faltantes}")

    df = df[colunas]
    df.columns = COLUNAS_PADRAO

    # Linhas totalmente vazias (rodapé ';;;;;' dos arquivos anuais).
    vazias = df["data_hora"].str.strip().eq("")
    ocorrencias["linhas_vazias_removidas"] += int(vazias.sum())
    return df.loc[~vazias].copy()


def padroniza(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica os tratamentos de qualidade, contabilizando cada um."""
    df["data_hora"] = pd.to_datetime(
        df["data_hora"].str.strip(), format="%d/%m/%Y %H:%M", errors="coerce"
    )
    ocorrencias["datas_invalidas"] += int(df["data_hora"].isna().sum())
    df = df.loc[df["data_hora"].notna()].copy()

    df["especialidade"] = df["especialidade"].map(normaliza_texto)

    df["municipio"] = df["municipio"].map(normaliza_texto)
    sem_municipio = df["municipio"].eq("")
    # Nota 3.3: 8,7% das consultas não têm município. Vira categoria explícita
    # em vez de exclusão — descartar 68 mil registros enviesaria as contagens
    # por especialidade, que não dependem do município.
    ocorrencias["municipio_nao_informado"] += int(sem_municipio.sum())
    df.loc[sem_municipio, "municipio"] = NAO_INFORMADO

    sexo_bruto = df["sexo"].map(normaliza_texto)
    df["sexo"] = sexo_bruto.map(MAPA_SEXO)
    sexo_desconhecido = df["sexo"].isna()
    ocorrencias["sexo_nao_informado"] += int(sexo_desconhecido.sum())
    df.loc[sexo_desconhecido, "sexo"] = NAO_INFORMADO

    idade = pd.to_numeric(df["idade"].str.strip(), errors="coerce")
    ocorrencias["idade_nao_numerica"] += int(idade.isna().sum())
    implausivel = idade > IDADE_MAXIMA_PLAUSIVEL
    ocorrencias["idade_implausivel"] += int(implausivel.fillna(False).sum())
    idade = idade.mask(implausivel)
    df["idade"] = idade.astype("Int64")

    return df


def main() -> int:
    arquivos = sorted(p for p in BRUTO.glob("*.csv") if p.name[:2] in {"05", "06"})
    if not arquivos:
        print(f"Nenhum CSV encontrado em {BRUTO}", file=sys.stderr)
        return 1

    partes = []
    for caminho in arquivos:
        tipo = "consulta" if caminho.name.startswith("05") else "internacao"
        df = le_arquivo(caminho)
        df["tipo"] = tipo
        df["arquivo_origem"] = caminho.name
        partes.append(df)
        print(f"  {caminho.name[:58]:<58} {len(df):>7} registros")

    df = padroniza(pd.concat(partes, ignore_index=True))
    df = df.sort_values("data_hora").reset_index(drop=True)

    PROCESSADO.mkdir(parents=True, exist_ok=True)
    destino = PROCESSADO / "atendimentos.csv.gz"
    df.to_csv(destino, index=False, compression="gzip")

    print(f"\nGravado: {destino.relative_to(RAIZ)}  ({len(df):,} registros)".replace(",", "."))
    print(f"Período: {df['data_hora'].min():%d/%m/%Y} a {df['data_hora'].max():%d/%m/%Y}")

    print("\nPor tipo de atendimento:")
    for tipo, n in df["tipo"].value_counts().items():
        print(f"  {tipo:<12} {n:>8,}".replace(",", "."))

    print("\nTratamentos aplicados (evidência para a seção 4.2):")
    for chave, n in sorted(ocorrencias.items()):
        print(f"  {chave:<30} {n:>8,}".replace(",", "."))

    print("\nCardinalidade das dimensões do painel:")
    for coluna in ("especialidade", "municipio", "sexo"):
        print(f"  {coluna:<15} {df[coluna].nunique():>5} categorias")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
