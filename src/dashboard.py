"""
Interface de apoio à decisão (CRISP-DM — Fase 6, Implantação).

Atende a quatro requisitos de informação:

    1. demanda histórica;
    2. especialidades com maior demanda;
    3. filtros pelas características disponíveis na base;
    4. consultas e internações previstas pelos modelos.

Duas regras estruturam a tela:

    · Observado é linha cheia azul; previsto é linha tracejada âmbar. A
      distinção NÃO depende de cor, então sobrevive a daltonismo e à
      impressão em preto e branco.
    · Onde não há sinal, a interface diz que não há — não inventa número.

Uso:
    ./venv/bin/streamlit run src/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
PROCESSADO = RAIZ / "data" / "processed"
TABELAS = RAIZ / "outputs" / "tabelas"

FAIXAS = ["0-14", "15-29", "30-44", "45-59", "60-74", "75+"]
MODELOS = ["Random Forest", "Gradient Boosting", "Poisson", "Baseline"]

AZUL = "#1E40AF"
AMBAR = "#B45309"
CINZA = "#64748B"

st.set_page_config(page_title="Painel de Demanda HUOL", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@300;400;500;600;700&family=Fira+Code:wght@400;500;600&display=swap');
      html, body, [class*="css"] { font-family: 'Fira Sans', sans-serif; }
      [data-testid="stMetricValue"] { font-family: 'Fira Code', monospace; font-size: 1.6rem; }
      [data-testid="stMetricLabel"] { font-size: .78rem; color: #64748B; }
      .rotulo { font-family: 'Fira Code', monospace; font-size: .68rem; letter-spacing: .1em;
                text-transform: uppercase; color: #1E40AF; }
      .nota { font-size: .82rem; color: #64748B; border-left: 3px solid #B45309;
              padding: .4rem .8rem; margin: .3rem 0 .8rem; }
      div[data-testid="stDataFrame"] { font-variant-numeric: tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Carregando atendimentos...")
def carrega_atendimentos() -> pd.DataFrame:
    df = pd.read_csv(PROCESSADO / "atendimentos.csv.gz", parse_dates=["data_hora"])
    df = df.loc[df["idade"].notna()].copy()
    df["periodo"] = df["data_hora"].dt.to_period("M").dt.to_timestamp()
    df["faixa"] = pd.cut(df["idade"], bins=[-1, 14, 29, 44, 59, 74, 200], labels=FAIXAS)
    return df.groupby(
        ["tipo", "periodo", "especialidade", "faixa", "municipio"], observed=True
    ).size().rename("n").reset_index()


@st.cache_data
def carrega_tabela(nome: str) -> pd.DataFrame:
    caminho = TABELAS / nome
    if not caminho.exists():
        return pd.DataFrame()
    df = pd.read_csv(caminho)
    if "periodo" in df.columns:
        df["periodo"] = pd.to_datetime(df["periodo"])
    return df


hist = carrega_atendimentos()
futuro = carrega_tabela("previsao_futura.csv")
backtest = carrega_tabela("previsoes.csv")
met_agreg = carrega_tabela("metricas_agregado.csv")
met_estrato = carrega_tabela("metricas_estrato.csv")


# --------------------------------------------------------------------------
# Filtros
# --------------------------------------------------------------------------

st.sidebar.markdown('<p class="rotulo">Filtros</p>', unsafe_allow_html=True)

rotulo_tipo = st.sidebar.radio(
    "Tipo de atendimento", ["Consultas ambulatoriais", "Internações hospitalares"]
)
tipo = "consulta" if rotulo_tipo.startswith("Consultas") else "internacao"

h_tipo = hist[hist["tipo"] == tipo]
ordem_esp = h_tipo.groupby("especialidade")["n"].sum().sort_values(ascending=False)

especialidades = st.sidebar.multiselect(
    "Especialidade",
    options=list(ordem_esp.index),
    default=list(ordem_esp.head(8).index),
    help="Vazio mostra todas as especialidades.",
)
if not especialidades:
    especialidades = list(ordem_esp.index)

faixas = st.sidebar.multiselect("Faixa etária", options=FAIXAS, default=FAIXAS)
if not faixas:
    faixas = FAIXAS

municipios_ordenados = h_tipo.groupby("municipio")["n"].sum().sort_values(ascending=False)
municipios = st.sidebar.multiselect(
    "Município de residência",
    options=list(municipios_ordenados.index),
    default=[],
    help="Vazio mostra todos os municípios.",
)
# Contagem e percentual calculados por tipo: o município ausente é 8,7% nas
# consultas e 0,02% nas internações, então um valor fixo erra num dos dois.
n_sem_municipio = municipios_ordenados.get("NAO INFORMADO", 0)
pct_sem_municipio = 100 * n_sem_municipio / municipios_ordenados.sum()
n_municipios = len(municipios_ordenados) - (n_sem_municipio > 0)
texto_pct = (
    "menos de 0,1%" if 0 < pct_sem_municipio < 0.1
    else f"{pct_sem_municipio:.1f}%".replace(".", ",")
)
st.sidebar.caption(
    f"{n_municipios} municípios · {texto_pct} dos registros sem informação, "
    "mantidos como categoria própria."
)

modelo = st.sidebar.selectbox("Modelo de previsão", MODELOS, index=0)
st.sidebar.caption("O seletor permite comparar os modelos avaliados no trabalho.")


def filtra(df: pd.DataFrame, com_municipio: bool) -> pd.DataFrame:
    m = df["especialidade"].isin(especialidades) & df["faixa"].isin(faixas)
    if com_municipio and municipios:
        m &= df["municipio"].isin(municipios)
    return df[m]


hist_f = filtra(h_tipo, com_municipio=True)
serie = hist_f.groupby("periodo")["n"].sum().sort_index()

fut_f = futuro[futuro["tipo"] == tipo] if not futuro.empty else pd.DataFrame()
if not fut_f.empty:
    fut_f = filtra(fut_f[fut_f["modelo"] == modelo], com_municipio=False)


# --------------------------------------------------------------------------
# Cabeçalho e indicadores
# --------------------------------------------------------------------------

st.markdown('<p class="rotulo">HUOL · Planejamento de capacidade</p>', unsafe_allow_html=True)
st.title("Painel de demanda por serviços hospitalares")

if serie.empty:
    st.warning(
        "Nenhum atendimento corresponde a esta combinação de filtros. "
        "Tente incluir mais especialidades, faixas etárias ou municípios."
    )
    st.stop()

ultimo_mes = serie.index.max()
ultimo_valor = int(serie.loc[ultimo_mes])
previsto = float(fut_f["predito"].sum()) if not fut_f.empty else None
mes_previsto = fut_f["periodo"].iloc[0] if not fut_f.empty else None

anos = serie.groupby(serie.index.year).mean()
variacao = (anos.get(2025, 0) / anos.get(2024, 1) - 1) * 100 if 2024 in anos and 2025 in anos else None

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Observado em {ultimo_mes:%m/%Y}", f"{ultimo_valor:,}".replace(",", "."))
c2.metric(
    "Média mensal 2025",
    f"{anos.get(2025, 0):,.0f}".replace(",", "."),
    f"{variacao:+.1f}% sobre 2024" if variacao is not None else None,
)
if previsto is not None:
    c3.metric(
        f"Previsto para {mes_previsto:%m/%Y}",
        f"{previsto:,.0f}".replace(",", "."),
        f"{100 * (previsto / ultimo_valor - 1):+.1f}% sobre o último mês",
    )
else:
    c3.metric("Previsto", "—", "execute previsao_futura.py")
c4.metric("Especialidades no filtro", f"{len(especialidades)}")


# --------------------------------------------------------------------------
# 1. Demanda histórica e previsão
# --------------------------------------------------------------------------

st.subheader("Demanda histórica e previsão")
st.caption(
    "A linha tracejada cobre os 6 meses de teste — onde o acerto do modelo pode ser "
    f"conferido contra o observado — e avança um mês, até {mes_previsto:%m/%Y}."
    if mes_previsto is not None else ""
)

# O tracejado cobre os 6 meses de teste (onde o acerto do modelo pode ser
# conferido contra o observado) e segue para o mês previsto. Mostrar só o
# ponto futuro esconderia justamente a evidência de que o modelo funciona.
linha_prev = pd.Series(dtype=float)
if not backtest.empty:
    bt = backtest[(backtest["tipo"] == tipo) & (backtest["modelo"] == modelo)]
    bt = filtra(bt, com_municipio=False)
    linha_prev = bt.groupby("periodo")["predito"].sum().sort_index()
if previsto is not None:
    linha_prev.loc[mes_previsto] = previsto
    linha_prev = linha_prev.sort_index()

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=serie.index, y=serie.values, name="Observado (linha cheia)",
        mode="lines", line=dict(color=AZUL, width=2.4),
        hovertemplate="%{x|%b/%Y}<br>%{y:,.0f} atendimentos<extra></extra>",
    )
)
if not linha_prev.empty:
    fig.add_trace(
        go.Scatter(
            x=linha_prev.index, y=linha_prev.values,
            name=f"{modelo} (linha tracejada)", mode="lines+markers",
            line=dict(color=AMBAR, width=2.4, dash="dash"),
            marker=dict(size=5),
            hovertemplate="%{x|%b/%Y}<br>%{y:,.0f} previstos<extra></extra>",
        )
    )
    fig.add_vline(x=ultimo_mes, line_width=1, line_dash="dot", line_color=CINZA)
    fig.add_annotation(
        x=ultimo_mes, y=1, yref="paper", yanchor="bottom", showarrow=False,
        text="  fim do observado", font=dict(size=10, color=CINZA), xanchor="left",
    )
fig.update_layout(
    height=380, margin=dict(l=10, r=10, t=10, b=10),
    hovermode="x unified", template="simple_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    yaxis_title="Atendimentos por mês", xaxis_title=None,
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with st.expander("Ver os dados desta série em tabela"):
    st.dataframe(
        serie.rename("atendimentos").reset_index().assign(
            periodo=lambda d: d["periodo"].dt.strftime("%m/%Y")
        ),
        use_container_width=True, hide_index=True,
    )


# --------------------------------------------------------------------------
# 2. Especialidades com maior demanda
# --------------------------------------------------------------------------

st.subheader("Especialidades com maior demanda")
janela = ultimo_mes - pd.DateOffset(months=11)
st.caption(f"Acumulado de {janela:%m/%Y} a {ultimo_mes:%m/%Y}")

top = (
    hist_f[hist_f["periodo"] >= janela]
    .groupby("especialidade")["n"].sum().sort_values(ascending=False).head(12)
)
fig2 = go.Figure(
    go.Bar(
        x=top.values[::-1], y=[e.title() for e in top.index[::-1]],
        orientation="h", marker_color=AZUL,
        hovertemplate="%{y}<br>%{x:,.0f} atendimentos<extra></extra>",
    )
)
fig2.update_layout(
    height=max(280, 26 * len(top)), margin=dict(l=10, r=10, t=10, b=10),
    template="simple_white", xaxis_title="Atendimentos em 12 meses", yaxis_title=None,
)
st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# --------------------------------------------------------------------------
# 3. Previsão por especialidade
# --------------------------------------------------------------------------

if not fut_f.empty:
    st.subheader(f"Previsão por especialidade — {mes_previsto:%m/%Y}")

    if municipios:
        st.markdown(
            '<div class="nota">Os filtros de município valem para o histórico, mas não para a '
            "previsão. O município foi retirado da chave do painel porque o recorte de cinco "
            "dimensões chegava a 98,2% de células vazias — decisão registrada na seção 4.2. "
            "A previsão abaixo considera todos os municípios.</div>",
            unsafe_allow_html=True,
        )

    obs_ultimo = (
        hist_f[hist_f["periodo"] == ultimo_mes].groupby("especialidade")["n"].sum()
    )
    prev_esp = fut_f.groupby("especialidade").agg(
        previsto=("predito", "sum"),
        sem_sinal=("estrato", lambda s: (s == "baixo").all()),
    )
    tabela = prev_esp.join(obs_ultimo.rename("observado"), how="left").fillna({"observado": 0})
    tabela["variacao"] = (tabela["previsto"] / tabela["observado"].replace(0, pd.NA) - 1) * 100
    tabela = tabela.sort_values("previsto", ascending=False).reset_index()

    # Onde o grupo inteiro está no estrato de baixo volume, nenhum modelo supera
    # a persistência (99,4% de zeros nas internações). Dizer "sem sinal" é mais
    # honesto do que exibir um número que não se sustenta.
    tabela["Previsto"] = [
        "sem sinal" if r.sem_sinal else f"{r.previsto:,.0f}".replace(",", ".")
        for r in tabela.itertuples()
    ]
    tabela["Variação"] = [
        "—" if r.sem_sinal or pd.isna(r.variacao) else f"{r.variacao:+.1f}%"
        for r in tabela.itertuples()
    ]
    tabela["Especialidade"] = tabela["especialidade"].str.title()
    tabela[f"Observado {ultimo_mes:%m/%Y}"] = tabela["observado"].map(
        lambda v: f"{v:,.0f}".replace(",", ".")
    )

    st.dataframe(
        tabela[["Especialidade", f"Observado {ultimo_mes:%m/%Y}", "Previsto", "Variação"]],
        use_container_width=True, hide_index=True, height=420,
    )


# --------------------------------------------------------------------------
# 4. Apoio à decisão — previsão confrontada com a capacidade planejada (PI3)
# --------------------------------------------------------------------------
#
# A previsão sozinha não decide nada. O gestor precisa saber: (i) se a demanda
# esperada cabe na capacidade que ele já planejou; (ii) se a diferença é maior
# do que o erro que o modelo costuma cometer naquela especialidade; e (iii) se
# naquela especialidade o modelo de fato superou a persistência no backtest.
# A tela responde às três perguntas, especialidade por especialidade, e
# classifica cada uma em uma ação sugerida para EXAME — não em uma ordem.

def classifica(lacuna: float, erro: float, confiavel: bool, sem_sinal: bool) -> str:
    """Regra de triagem, deliberadamente simples e explicável ao gestor.

    · sem sinal ................ estrato baixo: não há previsão a confrontar;
    · lacuna ≤ erro histórico .. a diferença está dentro do erro típico do
                                 modelo — não justifica mudar o plano;
    · lacuna > erro histórico .. a diferença supera o erro típico — vale
                                 examinar ampliação (ou redução) da agenda;
    · modelo não supera a persistência na especialidade: a sugestão vem com
      ressalva, porque o próprio histórico recente é tão bom quanto o modelo.
    """
    if sem_sinal:
        return "Sem sinal — decidir pelo histórico"
    if abs(lacuna) <= erro:
        acao = "Manter — diferença dentro do erro"
    elif lacuna > 0:
        acao = "Examinar ampliação"
    else:
        acao = "Examinar redução"
    return acao if confiavel else f"{acao} (baixa confiança)"


val_esp = carrega_tabela("validacao_por_especialidade.csv")

if not fut_f.empty and not val_esp.empty:
    st.subheader(f"Apoio à decisão — capacidade planejada × demanda prevista ({mes_previsto:%m/%Y})")
    st.markdown(
        '<div class="nota">Informe na coluna <b>Capacidade planejada</b> quantos atendimentos a '
        "especialidade está preparada para oferecer no próximo mês. O valor inicial é o "
        "realizado no último mês — o critério que o gestor usaria sem modelo. A ferramenta "
        "não decide: compara a previsão com o plano, mostra o erro que o modelo cometeu "
        "naquela especialidade nos seis meses de teste e indica onde a diferença merece "
        "exame.</div>",
        unsafe_allow_html=True,
    )

    v = val_esp[val_esp["tipo"] == tipo]
    erro_esp = v[v["modelo"] == modelo].set_index("especialidade")
    base_esp = v[v["modelo"] == "Baseline"].set_index("especialidade")["MAE"]

    plano = tabela[["especialidade", "Especialidade", "observado", "previsto", "sem_sinal"]].copy()
    plano["Capacidade planejada"] = plano["observado"].round().astype(int)
    plano["erro"] = plano["especialidade"].map(erro_esp["MAE"])
    plano["confiavel"] = plano["especialidade"].map(erro_esp["MAE"] <= base_esp)

    editado = st.data_editor(
        plano[["Especialidade", "Capacidade planejada"]],
        column_config={
            "Especialidade": st.column_config.TextColumn(disabled=True),
            "Capacidade planejada": st.column_config.NumberColumn(min_value=0, step=1),
        },
        use_container_width=True, hide_index=True, height=260, key=f"plano_{tipo}_{modelo}",
    )
    plano["Capacidade planejada"] = editado["Capacidade planejada"].to_numpy()
    plano["lacuna"] = plano["previsto"] - plano["Capacidade planejada"]
    plano["Ação sugerida"] = [
        classifica(r.lacuna, r.erro, bool(r.confiavel), bool(r.sem_sinal))
        for r in plano.itertuples()
    ]

    def fmt(x: float) -> str:
        return f"{x:,.0f}".replace(",", ".")

    resultado = pd.DataFrame({
        "Especialidade": plano["Especialidade"],
        "Capacidade planejada": plano["Capacidade planejada"].map(fmt),
        "Previsto": [("—" if s else fmt(p)) for p, s in zip(plano["previsto"], plano["sem_sinal"])],
        "Faixa provável": [
            "—" if s else f"{fmt(max(p - e, 0))} a {fmt(p + e)}"
            for p, e, s in zip(plano["previsto"], plano["erro"], plano["sem_sinal"])
        ],
        "Lacuna (prev. − plano)": [
            "—" if s else f"{l:+,.0f}".replace(",", ".")
            for l, s in zip(plano["lacuna"], plano["sem_sinal"])
        ],
        "Ação sugerida": plano["Ação sugerida"],
    })

    examinar = plano["Ação sugerida"].str.startswith("Examinar")
    k1, k2, k3 = st.columns(3)
    k1.metric("Especialidades a examinar", f"{int(examinar.sum())}")
    k2.metric(
        "Demanda prevista acima do plano",
        fmt(plano.loc[examinar & (plano["lacuna"] > 0), "lacuna"].sum()),
        help="Soma das lacunas positivas que superam o erro histórico.",
    )
    k3.metric(
        "Especialidades com baixa confiança",
        f"{int((~plano['confiavel'].fillna(False).astype(bool) & ~plano['sem_sinal'].astype(bool)).sum())}",
        help="O modelo não superou a persistência nessa especialidade no backtest.",
    )

    st.dataframe(
        resultado.sort_values("Ação sugerida"), use_container_width=True,
        hide_index=True, height=420,
    )
    st.caption(
        "Faixa provável = previsão ± erro médio absoluto (MAE) do modelo naquela especialidade "
        "nos seis meses de teste. A faixa é empírica e não é intervalo de confiança: indica a "
        "magnitude típica do erro, não uma probabilidade. Previsão de um mês à frente apenas."
    )

    # Ficha de uma especialidade: o raciocínio completo, em linguagem de gestão.
    escolha = st.selectbox("Detalhar especialidade", plano["Especialidade"].tolist())
    r = plano[plano["Especialidade"] == escolha].iloc[0]
    if r.sem_sinal:
        st.info(
            f"**{escolha}** está inteiramente no estrato de baixo volume. Nenhum modelo superou "
            "a persistência nesse regime; a decisão deve seguir o histórico recente."
        )
    else:
        ganho = 100 * (1 - r.erro / base_esp.get(r.especialidade, float("nan")))
        dec = lambda x, f: format(x, f).replace(".", ",")  # decimal em pt-BR
        lacuna_txt = f"{r.lacuna:+,.0f}".replace(",", ".")
        st.markdown(
            f"- Realizado no último mês: **{fmt(r.observado)}** · capacidade planejada: "
            f"**{fmt(r['Capacidade planejada'])}**\n"
            f"- Previsão ({modelo}) para {mes_previsto:%m/%Y}: **{fmt(r.previsto)}** — "
            f"faixa provável de {fmt(max(r.previsto - r.erro, 0))} a {fmt(r.previsto + r.erro)}\n"
            f"- Erro histórico do modelo nesta especialidade: **{dec(r.erro, '.1f')}** "
            f"atendimentos/mês ({dec(ganho, '+.1f')}% em relação à persistência)\n"
            f"- Lacuna projetada: **{lacuna_txt}** atendimentos → **{r['Ação sugerida']}**"
        )
        st.caption(
            "Pergunta gerencial: a lacuna justifica rever a agenda? A ferramenta não responde "
            "sozinha — fornece a magnitude da diferença, o erro esperado e a confiabilidade do "
            "modelo nesta especialidade para que o gestor examine a decisão."
        )


# --------------------------------------------------------------------------
# 5. Desempenho do modelo — a evidência por trás do número
# --------------------------------------------------------------------------

st.subheader("Desempenho dos modelos")
st.caption(
    "Erro medido por origem rolante nos 6 meses de teste (jan–jun/2026), "
    "com as previsões reagregadas para especialidade-mês."
)

col_a, col_b = st.columns([1, 1])

if not met_agreg.empty:
    m = met_agreg[met_agreg["tipo"] == tipo].sort_values("MAE")
    base_mae = m.loc[m["modelo"] == "Baseline", "MAE"].iloc[0]
    m = m.assign(
        Ganho=lambda d: [
            "—" if r.modelo == "Baseline" else f"{100 * (1 - r.MAE / base_mae):+.1f}%"
            for r in d.itertuples()
        ]
    )
    col_a.markdown("**Comparação geral**")
    col_a.dataframe(
        m[["modelo", "MAE", "RMSE", "Ganho"]].rename(columns={"modelo": "Modelo"}).round(2),
        use_container_width=True, hide_index=True,
    )

if not met_estrato.empty:
    e = met_estrato[met_estrato["tipo"] == tipo]
    melhores = (
        e.loc[e.groupby("estrato")["MAE"].idxmin()][["estrato", "modelo", "MAE"]]
        .rename(columns={"estrato": "Estrato", "modelo": "Melhor modelo"})
    )
    col_b.markdown("**Melhor modelo por estrato de volume**")
    col_b.dataframe(melhores.round(2), use_container_width=True, hide_index=True)
    col_b.caption(
        "No estrato de baixo volume nenhum modelo supera a persistência: "
        "não há sinal a extrair."
    )

st.divider()
st.caption(
    "Importância de variável, quando exibida, expressa **relevância preditiva** — "
    "a contribuição para a capacidade de previsão do modelo. Não indica relação de "
    "causa e efeito. · Dados públicos EBSERH/HUOL-UFRN, jan/2023 a jun/2026."
)
