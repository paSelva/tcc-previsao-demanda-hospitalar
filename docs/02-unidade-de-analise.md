# Unidade de análise e construção do painel (CRISP-DM — Fase 3)

Complementa a seção 4.2 do TCC. Gerado por `src/painel.py`, que grava
`data/processed/painel.csv.gz`.

## Escolha da chave do painel

A chave foi escolhida depois de medir a esparsidade de cada recorte candidato. A tabela
mostra a proporção de células vazias na grade completa, com agregação mensal.

| Recorte | Consultas | Internações |
|---|---:|---:|
| mês × especialidade | 7,4% | 21,3% |
| **mês × especialidade × faixa etária** | **42,9%** | **62,4%** |
| mês × especialidade × faixa × sexo | 58,6% | 81,0% |
| mês × especialidade × município (20 maiores) | 36,9% | 81,0% |
| mês × especialidade × município × faixa × sexo | 89,4% | 98,2% |
| semana × especialidade × município × faixa × sexo | 95,1% | 99,4% |

**Decisão: `mês × especialidade × faixa etária`, igual nas duas bases**, para que a
comparação entre consultas e internações seja legítima. Sexo e município voltam como
composição do grupo, sempre medida no mês anterior.

A proporção de zeros não justifica, por si, a Regressão de Poisson. As premissas do modelo
foram testadas à parte (seção 4.3.3 do TCC e `outputs/tabelas/validacao_poisson_*.csv`).

## Painel resultante

| | Consultas | Internações |
|---|---:|---:|
| Linhas | 12.600 | 9.000 |
| Meses modeláveis | 30 (jan/2024 – jun/2026) | 30 |
| Especialidades | 70 | 50 |
| Treino / teste (linhas) | 10.080 / 2.520 | 7.200 / 1.800 |

As linhas são séries de especialidade × faixa acompanhadas nos **mesmos 30 meses**, e não
observações temporais independentes. O ano de 2023 serve apenas para gerar as defasagens de
12 meses. O teste corresponde aos meses de janeiro a junho de 2026.

## Preditores

| Grupo | Variáveis |
|---|---|
| Defasagens do alvo | `lag_1`, `lag_2`, `lag_3`, `lag_12` |
| Médias móveis defasadas | `media_3m`, `media_6m` |
| Composição do grupo em t−1 | `idade_media_lag1`, `prop_feminino_lag1`, `prop_natal_lag1`, `prop_mun_ni_lag1`, `n_municipios_lag1` |
| Calendário | `mes`, `ano`, `trimestre`, `dias_uteis` |

## Controle de vazamento temporal

- Defasagens: `groupby(...).shift(d)` dentro de cada grupo.
- Médias móveis: a série é deslocada um mês antes da média (`shift(1).rolling(j)`).
- Imputação da composição e definição dos estratos de volume: calculadas só com o treino.
- Grade completa com zeros explícitos: um grupo sem atendimento tem contagem zero, não dado
  ausente.
- Divisão treino/teste por período, nunca por linha. Os modelos são reajustados em cada uma
  das seis origens de previsão (`src/modelagem.py`).
