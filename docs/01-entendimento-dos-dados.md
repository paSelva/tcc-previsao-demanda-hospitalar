# Entendimento dos dados (CRISP-DM — Fase 2)

Complementa a seção 4.1 do TCC. O tratamento de cada problema é aplicado por
`src/ingestao.py`, que imprime a contagem de registros afetados.

## Fontes

| Base | Conjunto no portal | Responsável |
|---|---|---|
| Consultas ambulatoriais | https://dados.gov.br/dados/conjuntos-dados/05-consultas-ambulatoriais | HUOL/UFRN — EBSERH |
| Internações hospitalares | https://dados.gov.br/dados/conjuntos-dados/06-internacoes-hospitalares | HUOL/UFRN — EBSERH |

As duas bases são **microdados**, com uma linha por atendimento e sem identificador de paciente:

```
consultas   : data_consulta; especialidade; município; idade; sexo
internações : data_internacao; especialidade; município; idade; sexo
```

## Cobertura e volume

| Período de referência | Consultas | Internações |
|---|---:|---:|
| Ano de 2023 | 212.193 | 6.062 |
| jan–mar 2024 | 49.727 | 1.642 |
| abr–jun 2024 | 57.914 | 1.783 |
| jul–set 2024 | 61.368 | 2.110 |
| out–dez 2024 | 56.254 | 2.044 |
| jan–mar 2025 | 53.665 | 1.943 |
| abr–jun 2025 *(publicado como "MarAbrMai")* | 60.870 | 2.169 |
| jul–set 2025 | 66.460 | 2.274 |
| out–dez 2025 | 59.629 | 2.081 |
| jan–mar 2026 | 48.505 | 1.919 |
| abr–jun 2026 | 51.676 | 2.001 |
| **Total** | **778.261** | **26.028** |

A cobertura vai de 01/01/2023 a 30/06/2026, em períodos contínuos e sem sobreposição.

## Problemas de qualidade e tratamento

| # | Problema | Tratamento |
|---|---|---|
| 1 | O trimestre out–dez/2025 de consultas foi publicado em CSV e também em PDF | Usa-se o CSV; o PDF é ignorado |
| 2 | Codificação de `sexo` varia entre arquivos (`F`/`FEMININO`, `I`/`INFEFINIDO`/`INDEFINIDO`) | Mapeamento para `FEMININO`, `MASCULINO`, `INDEFINIDO` |
| 3 | `município` ausente em 8,7% das consultas | Categoria própria "NAO INFORMADO" |
| 4 | Um arquivo de consultas (abr–jun/2024) tem três colunas extras | Colunas descartadas para manter o esquema comum |
| 5 | Os arquivos anuais de 2023 têm preâmbulo de metadados e cabeçalho diferente | Leitor trata os dois formatos |
| 6 | Arquivos em ISO-8859-1 com quebras CRLF | Conversão para UTF-8 |
| 7 | O recurso "MarAbrMai de 2025" contém, na verdade, dados de abril a junho | Período definido pelo conteúdo, não pelo rótulo |
| 8 | `idade` não numérica ou implausível (até 124 anos) | Registros excluídos do painel (55 no total) |
| 9 | Grafia de `município` com caixa inconsistente | Normalização de caixa e acentos |

## Perfil das bases

- 70 especialidades de consulta e 50 de internação; 46 em comum.
- Oftalmologia lidera as consultas; Cardiologia lidera as internações.
- Natal responde por 34,6% das internações; a cauda de municípios é longa e esparsa.
