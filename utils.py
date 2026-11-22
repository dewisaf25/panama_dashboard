"""Utilitas bersama: loader data, tema, komponen UI."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
DATA, ASSETS = ROOT / "data", ROOT / "assets"

# ---------------------------------------------------------------- konstanta
SPLITS = {  # sesuai notebook (70 : 10 : 20, kronologis)
    "Train (70%)":    ("2015-01-03", "2018-11-03", "#6366F1"),
    "Validasi (10%)": ("2018-11-04", "2019-05-22", "#F59E0B"),
    "Test (20%)":     ("2019-05-23", "2020-06-26", "#10B981"),
}
TEST_START, TEST_END = pd.Timestamp("2019-05-23"), pd.Timestamp("2020-06-26")
COVID_START = pd.Timestamp("2020-03-15")
ANOMALY_THR = 600

DOW_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

BEST = "LightGBM (ML)"
NAIVE = "Naive-168 (minggu lalu)"
RESULT_FILES = {"Test total": "hasil_test.csv", "Pra-COVID": "hasil_test_pra_covid.csv",
                "Saat COVID": "hasil_test_covid.csv"}


# ---------------------------------------------------------------- tampilan
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1E1B4B 0%, #312E81 100%); }
[data-testid="stSidebar"] * { color: #E0E7FF !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: rgba(255,255,255,.14); border-radius: 10px; }
.hero { background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #EC4899 120%); border-radius: 20px;
        padding: 34px 38px; color: #fff; margin-bottom: 22px; box-shadow: 0 10px 30px rgba(79,70,229,.25); }
.hero h1 { color: #fff; font-size: 2.1rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.5px; }
.hero p { color: #E0E7FF; font-size: 1.02rem; margin: 0; max-width: 820px; line-height: 1.55; }
.hero .tag { display:inline-block; background: rgba(255,255,255,.18); padding: 3px 12px; border-radius: 999px;
             font-size: .78rem; font-weight: 600; margin: 14px 8px 0 0; }
.phead { margin-bottom: 18px; }
.phead h2 { font-weight: 800; font-size: 1.7rem; color: #1E1B4B; margin: 0; letter-spacing: -0.4px; }
.phead p { color: #64748B; margin: 4px 0 0 0; font-size: .98rem; }
.kpi { background: #fff; border: 1px solid #E2E8F0; border-left: 5px solid var(--accent, #4F46E5); border-radius: 14px;
       padding: 16px 18px; box-shadow: 0 2px 10px rgba(15,23,42,.05); height: 100%; }
.kpi-l { color: #64748B; font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }
.kpi-v { color: #0F172A; font-size: 1.75rem; font-weight: 800; line-height: 1.25; margin-top: 2px; }
.kpi-s { color: #64748B; font-size: .84rem; margin-top: 2px; }
.card { background: #fff; border: 1px solid #E2E8F0; border-radius: 16px; padding: 20px 22px; height: 100%;
        box-shadow: 0 2px 10px rgba(15,23,42,.05); color: #0F172A; }
.card h4 { margin: 0 0 8px 0; font-size: 1.02rem; font-weight: 700; color: #1E1B4B; }
.card p, .card li { color: #334155; font-size: .93rem; line-height: 1.55; margin: 0; }
.chip { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600; color:#fff; }
.note { background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 12px; padding: 12px 16px; color: #3730A3; font-size: .92rem; }
.warn { background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 12px; padding: 12px 16px; color: #92400E; font-size: .92rem; }
h3 { color: #1E1B4B; font-weight: 700; }
#MainMenu, footer { visibility: hidden; }
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title, subtitle, tags=()):
    chips = "".join(f'<span class="tag">{t}</span>' for t in tags)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{chips}</div>', unsafe_allow_html=True)


def page_header(title, subtitle=""):
    st.markdown(f'<div class="phead"><h2>{title}</h2><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(label, value, sub="", color="#4F46E5"):
    st.markdown(f'<div class="kpi" style="--accent:{color}"><div class="kpi-l">{label}</div>'
                f'<div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>', unsafe_allow_html=True)


def card(title, body, color="#4F46E5"):
    st.markdown(f'<div class="card" style="border-top:4px solid {color}"><h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True)


def note(text, kind="note"):
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


def chip(text, color):
    return f'<span class="chip" style="background:{color}">{text}</span>'


# ---------------------------------------------------------------- plotly helpers
def style_fig(fig, height=420, title=None, legend_bottom=True):
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=56 if title else 24, b=10),
        font=dict(family="Inter, sans-serif", size=13, color="#0F172A"),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=16, color="#1E1B4B")))
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="left", x=0))
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    return fig


def plot(fig, **kw):
    """Kompatibel lintas versi Streamlit (width='stretch' vs use_container_width)."""
    try:
        st.plotly_chart(fig, width="stretch", theme=None, **kw)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None, **kw)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def shade(fig, x0, x1, color, label=None, opacity=0.10):
    """Blok warna vertikal (hindari bug add_vrect+anotasi pada sumbu tanggal)."""
    fig.add_shape(type="rect", xref="x", yref="paper", x0=str(x0), x1=str(x1), y0=0, y1=1,
                  fillcolor=color, opacity=opacity, line_width=0, layer="below")
    if label:
        fig.add_annotation(x=str(x0), xref="x", y=1, yref="paper", text=label, showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=11, color=color))


def vline(fig, x, color="#EF4444", label=None):
    fig.add_shape(type="line", xref="x", yref="paper", x0=str(x), x1=str(x), y0=0, y1=1,
                  line=dict(color=color, width=1.6, dash="dash"))
    if label:
        fig.add_annotation(x=str(x), xref="x", y=1, yref="paper", text=label, showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=11, color=color))


# ---------------------------------------------------------------- data
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MAPE%": float(np.mean(np.abs(e / y)) * 100),
            "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))}


@st.cache_data(show_spinner=False)
def load_hourly():
    df = pd.read_csv(DATA / "hourly_data.csv.gz", parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.normalize()
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["T_mean"] = df[["T2M_toc", "T2M_san", "T2M_dav"]].mean(axis=1)
    df["weekend"] = df["dow"] >= 5
    df["nat_clean"] = df["nat_demand"].where(df["nat_demand"] >= ANOMALY_THR)   # anomali -> NaN
    return df


@st.cache_data(show_spinner=False)
def load_results():
    """Hasil resmi dari notebook (LightGBM + patokan Naive-168) untuk 3 periode."""
    return {k: pd.read_csv(DATA / f, index_col=0) for k, f in RESULT_FILES.items()}


@st.cache_data(show_spinner=False)
def load_importance():
    return pd.read_csv(DATA / "importance_variabel.csv", index_col=0)


def _naive_frame():
    h = load_hourly()
    y = h["nat_demand"].where(h["nat_demand"] >= ANOMALY_THR).interpolate(limit_direction="both")
    d = pd.DataFrame({"datetime": h["datetime"], "actual": h["nat_demand"], NAIVE: y.shift(168).values})
    return d[(d["datetime"] >= TEST_START) & (d["datetime"] < TEST_END + pd.Timedelta(days=1))].reset_index(drop=True)


@st.cache_data(show_spinner="Menyiapkan model LightGBM (hanya sekali, ± 30–60 detik)...")
def load_predictions():
    """Return (df, sumber). df: datetime, actual, LightGBM (ML), Naive-168.
    sumber: 'file' (hasil ekspor notebook) | 'trained' (dilatih di aplikasi) | 'baseline:<pesan>' (gagal)."""
    p = DATA / "dashboard_predictions.csv.gz"
    if p.exists():
        d = pd.read_csv(p, parse_dates=["datetime"])
        if BEST in d.columns:
            if NAIVE not in d.columns:
                d = d.merge(_naive_frame()[["datetime", NAIVE]], on="datetime", how="left")
            return d[["datetime", "actual", BEST, NAIVE]], "file"
    try:
        import forecasting
        return forecasting.train_and_predict(DATA / "hourly_data.csv.gz"), "trained"
    except Exception as e:  # noqa: BLE001
        return _naive_frame(), f"baseline:{type(e).__name__}: {str(e)[:150]}"
