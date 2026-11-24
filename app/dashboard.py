"""Tableau de bord interactif pour explorer les séries (or + covariables).

Lancement depuis la racine du dépôt :
    streamlit run app/dashboard.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

st.set_page_config(page_title="Or & covariables", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    gold = pd.read_csv(DATA_DIR / "gold_daily.csv", parse_dates=["Date"], index_col="Date")
    cov = pd.read_csv(DATA_DIR / "covariates_daily.csv", parse_dates=["Date"], index_col="Date")
    gold = gold.rename(columns={c: f"GOLD_{c}" for c in gold.columns})
    return gold.join(cov, how="outer").sort_index()


df = load_data()
gold_ohlc = df[["GOLD_Open", "GOLD_High", "GOLD_Low", "GOLD_Close"]].dropna()
series_cols = ["GOLD_Close", "DXY", "VIX", "TNX_10Y", "OIL", "SP500"]

# ---------------------------------------------------------------- barre latérale
st.sidebar.title("Paramètres")
dmin, dmax = df.index.min().date(), df.index.max().date()
period = st.sidebar.date_input("Période", value=(dmin, dmax), min_value=dmin, max_value=dmax)
start, end = (period[0], period[-1]) if isinstance(period, (tuple, list)) else (period, dmax)
selected = st.sidebar.multiselect("Séries", series_cols, default=["GOLD_Close"])
transform = st.sidebar.radio(
    "Transformation",
    ["Niveau", "Base 100", "Log-rendements", "Volatilité glissante (annualisée)"],
)
window = st.sidebar.slider("Fenêtre glissante (jours)", 5, 252, 21)
log_y = st.sidebar.checkbox("Échelle log (niveaux)", value=False)

sub = df.loc[str(start):str(end)]


def safe_log(x):
    """Log en ignorant les valeurs <= 0 (ex. pétrole négatif en avril 2020)."""
    return np.log(x.where(x > 0))


def transform_series(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if transform == "Base 100":
        return 100 * s / s.iloc[0]
    if transform == "Log-rendements":
        return safe_log(s).diff().dropna()
    if transform.startswith("Volatilité"):
        r = safe_log(s).diff()
        return (r.rolling(window).std() * np.sqrt(252)).dropna()
    return s


st.title("Exploration des séries : or (XAU/USD) et covariables")

tab_series, tab_gold, tab_dist, tab_corr = st.tabs(
    ["Courbes", "Or (chandeliers)", "Distribution des rendements", "Corrélations"]
)

# ---------------------------------------------------------------- onglet courbes
with tab_series:
    if not selected:
        st.info("Choisissez au moins une série dans la barre latérale.")
    else:
        separate = st.checkbox("Un graphique par série", value=len(selected) > 1 and transform == "Niveau")
        if separate:
            fig = make_subplots(rows=len(selected), cols=1, shared_xaxes=True,
                                subplot_titles=selected, vertical_spacing=0.04)
            for i, col in enumerate(selected, start=1):
                s = transform_series(sub[col])
                fig.add_trace(go.Scatter(x=s.index, y=s, name=col, line=dict(width=1)), row=i, col=1)
            fig.update_layout(height=260 * len(selected), showlegend=False)
        else:
            fig = go.Figure()
            for col in selected:
                s = transform_series(sub[col])
                fig.add_trace(go.Scatter(x=s.index, y=s, name=col, line=dict(width=1)))
            fig.update_layout(height=550)
            if transform == "Niveau" and len(selected) > 1:
                st.caption("Astuce : utilisez « Base 100 » pour comparer des séries d'échelles différentes.")
        if log_y and transform in ("Niveau", "Base 100"):
            fig.update_yaxes(type="log")
        fig.update_layout(hovermode="x unified", margin=dict(t=40, b=20))
        fig.update_xaxes(rangeslider_visible=not separate)
        st.plotly_chart(fig, width="stretch")

        stats = pd.DataFrame({c: transform_series(sub[c]).describe() for c in selected}).T
        st.subheader("Statistiques descriptives")
        st.dataframe(stats.style.format("{:.4f}"), width="stretch")

# ---------------------------------------------------------------- onglet or OHLC
with tab_gold:
    g = gold_ohlc.loc[str(start):str(end)]
    freq = st.selectbox("Agrégation", ["Jour", "Semaine", "Mois"], index=1)
    rule = {"Jour": None, "Semaine": "W", "Mois": "ME"}[freq]
    if rule:
        g = g.resample(rule).agg({"GOLD_Open": "first", "GOLD_High": "max",
                                  "GOLD_Low": "min", "GOLD_Close": "last"}).dropna()
    fig = go.Figure(go.Candlestick(x=g.index, open=g.GOLD_Open, high=g.GOLD_High,
                                   low=g.GOLD_Low, close=g.GOLD_Close, name="Or"))
    ma = g.GOLD_Close.rolling(window).mean()
    fig.add_trace(go.Scatter(x=ma.index, y=ma, name=f"Moyenne mobile {window}", line=dict(width=1.5)))
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, margin=dict(t=30, b=20))
    if log_y:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- onglet distribution
with tab_dist:
    col = st.selectbox("Série", series_cols, index=0, key="dist")
    r = safe_log(sub[col].dropna()).diff().dropna()
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.histogram(r, nbins=150, histnorm="probability density", labels={"value": "log-rendement"})
        x = np.linspace(r.min(), r.max(), 400)
        pdf = np.exp(-0.5 * ((x - r.mean()) / r.std()) ** 2) / (r.std() * np.sqrt(2 * np.pi))
        fig.add_trace(go.Scatter(x=x, y=pdf, name="Loi normale (même moyenne/écart-type)"))
        fig.update_layout(height=450, showlegend=True, margin=dict(t=30, b=20))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.metric("Moyenne journalière", f"{r.mean():.5f}")
        st.metric("Volatilité annualisée", f"{r.std() * np.sqrt(252):.2%}")
        st.metric("Asymétrie (skewness)", f"{r.skew():.3f}")
        st.metric("Kurtosis (excès)", f"{r.kurt():.3f}")

# ---------------------------------------------------------------- onglet corrélations
with tab_corr:
    rets = safe_log(sub[series_cols]).diff().dropna()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Corrélation des log-rendements")
        fig = px.imshow(rets.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=450, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Corrélation glissante avec l'or")
        other = st.selectbox("Comparer avec", [c for c in series_cols if c != "GOLD_Close"])
        rc = rets["GOLD_Close"].rolling(window).corr(rets[other]).dropna()
        fig = px.line(rc, labels={"value": "corrélation", "index": ""})
        fig.add_hline(y=0, line_dash="dot")
        fig.update_layout(height=450, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
