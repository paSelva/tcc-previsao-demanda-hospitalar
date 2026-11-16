# Análise preditiva da demanda por serviços hospitalares

Código, dados organizados e resultados do Trabalho de Conclusão de Curso *Análise preditiva
da demanda por serviços hospitalares para apoio ao planejamento: estudo de caso com dados
públicos de um hospital universitário* (Sistemas de Informação, UNI7, 2026).

O estudo de caso usa as bases públicas de consultas ambulatoriais e internações hospitalares
do Hospital Universitário Onofre Lopes (HUOL/UFRN), de janeiro de 2023 a junho de 2026.
Os modelos (Regressão de Poisson, Random Forest e Gradient Boosting) são comparados com duas
referências ingênuas (persistência e sazonal de 12 meses), e os resultados alimentam uma
interface de apoio à decisão em Streamlit.

## Mapa do CRISP-DM

| Fase | Script | Evidência gravada | Seção do TCC |
|---|---|---|---|
| 1. Entendimento do negócio | — | — | 3.2 |
| 2. Entendimento dos dados | `src/ingestao.py` | relatório no terminal; [`docs/01-entendimento-dos-dados.md`](docs/01-entendimento-dos-dados.md) | 4.1 |
| 3. Preparação dos dados | `src/painel.py` | `data/processed/painel.csv.gz`; [`docs/02-unidade-de-analise.md`](docs/02-unidade-de-analise.md) | 4.2 |
| 4. Modelagem | `src/modelagem.py` | `outputs/tabelas/previsoes.csv` | 4.3 |
| 5. Avaliação | `src/modelagem.py`, `src/validacao.py` | `outputs/tabelas/metricas_*.csv`, `outputs/tabelas/validacao_*.csv`, `outputs/validacao.log` | 4.4 |
| 6. Implantação | `src/previsao_futura.py`, `src/dashboard.py` | `outputs/tabelas/previsao_futura.csv`, `outputs/figuras/fig10` a `fig13` | 4.5 |

As figuras do Capítulo 4 são geradas por `src/figuras.py`.

## Estrutura

```
src/               scripts, na ordem das fases do CRISP-DM
data/raw/          bases originais (não versionadas; ver "Dados")
data/processed/    painel de contagens agregadas (versionado)
outputs/tabelas/   tabelas de resultado citadas no TCC
outputs/figuras/   figuras do TCC, incluindo as telas da interface
docs/              documentação das Fases 2 e 3
```

## Ambiente

Python 3.9. As versões exatas das bibliotecas estão em `requirements.txt`; outras versões
podem alterar as métricas.

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Reproduzir a modelagem e a avaliação (sem baixar as bases)

O painel agregado está versionado, então as Fases 4 e 5 rodam direto:

```bash
./venv/bin/python src/modelagem.py           # seis origens, quatro modelos, métricas
./venv/bin/python src/validacao.py           # testes complementares (~15 min)
./venv/bin/python src/validacao.py --rapido  # idem, sem a importância por permutação
```

Todas as fontes de aleatoriedade têm semente fixa, e a reexecução reproduz os números do TCC.

## Dados

Os microdados originais não são redistribuídos aqui. Para reproduzir as Fases 2, 3 e 6,
baixe as bases no portal (exige login gov.br) e salve os arquivos em `data/raw/`:

1. https://dados.gov.br/dados/conjuntos-dados/05-consultas-ambulatoriais
2. https://dados.gov.br/dados/conjuntos-dados/06-internacoes-hospitalares

Depois:

```bash
./venv/bin/python src/ingestao.py          # bases originais -> microdados padronizados
./venv/bin/python src/painel.py            # microdados -> painel de contagens
./venv/bin/python src/previsao_futura.py   # previsão do mês seguinte
./venv/bin/streamlit run src/dashboard.py  # interface de apoio à decisão
```

A previsão do mês seguinte e a interface leem os microdados padronizados
(`data/processed/atendimentos.csv.gz`), gerados pela ingestão.

## Nota sobre interpretação

A importância das variáveis expressa **relevância preditiva**, não relação de causa e efeito.
As previsões descrevem a demanda **atendida** pelo hospital, não a necessidade de saúde da
população.
