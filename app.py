# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Sovereign Dashboard v3  —  Streamlit  —  3 pestañas
#   Tab 1 : Dashboard Señales  (nuevo scoring v2)
#   Tab 2 : Gráficos Estrategia
#   Tab 3 : Mi Cartera
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import io
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st

from ta.trend    import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator

from indicators import (
    download_df, clean_yf_df,
    mcginley_dynamic, calculate_pvi, calculate_nvi,
    calc_mfi_blai5, calc_stoch, awesome_osc,
    calculate_bbwp, bbwp_signal,
    compute_blai5_koncorde, blai5_signals,
    clasificar_bitman,
    detectar_divergencia_simple,
    azul_z_score, calcular_velas_señal,
    semaforo, get_sovereign_dashboard,
    # nuevas funciones v2
    calcular_señales_v2,
    semaforo_salida,
    get_sovereign_dashboard_v2,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Sovereign Dashboard v3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    body, .stApp { background-color: #0d0f14; color: #ffffff; }
    .stDataFrame thead th { background: #13161e !important; color: #efb030 !important; }
    .stDataFrame tbody td { color: #ffffff !important; font-size: 0.78rem; }
    .stDataFrame tbody tr:hover { background: #1f2430 !important; }
    .stDataFrame { overflow-x: auto !important; }
    [data-testid="stMetricValue"]  { color: #ffffff !important; }
    [data-testid="stMetricDelta"]  { color: #c8cad0 !important; }
    [data-testid="stMetricLabel"]  { color: #efb030 !important; }
    label { color: #ffffff !important; }
    .stSlider label { color: #ffffff !important; }
    .stCaption { color: #aaaaaa !important; }

    div[data-baseweb="tab-list"] {
        background-color: #0d0f14 !important;
        border-bottom: 2px solid #efb030 !important;
        gap: 6px !important;
    }
    button[data-baseweb="tab"] {
        background-color: #1a1e2e !important;
        color: #aaaaaa !important;
        font-size: 1rem !important;
        font-weight: bold !important;
        font-family: monospace !important;
        letter-spacing: 0.05em !important;
        padding: 10px 28px !important;
        border-radius: 6px 6px 0px 0px !important;
        border: 1px solid #2a2e45 !important;
        border-bottom: none !important;
    }
    button[data-baseweb="tab"]:hover {
        background-color: #252a3a !important;
        color: #efb030 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #efb030 !important;
        color: #0d0f14 !important;
        border-color: #efb030 !important;
        font-size: 1.05rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"]::after {
        display: none !important;
    }
    @media (max-width: 768px) {
        .stDataFrame tbody td { font-size: 0.65rem !important; }
        h2 { font-size: 1.1rem !important; }
        [data-testid="stMetricValue"] { font-size: 1rem !important; }
        button[data-baseweb="tab"] { font-size: 0.8rem !important; padding: 8px 12px !important; }
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PALETA
# ══════════════════════════════════════════════════════════════════════════════

STYLE = dict(
    bg="#0d0f14",        panel="#13161e",      border="#1f2430",
    bull="#26a65b",      bear="#e04040",
    bull_fade="#26a65b55",                      bear_fade="#e0404055",
    mcg="#efb030",       ema200="#6060dd",
    text="#ffffff",      muted="#aaaaaa",
    verde="#2ca85e",     marron="#a06432",
    azul="#4488e0",      media_k="#ffffff",
    pvi="#6090e0",       pvi_ema="#efb030",
    macd_line="#6090e0", macd_sig="#efb030",
    rsi="#a78bfa",       adx="#ffffff",
    pdi="#26a65b",       ndi="#e04040",
    ao_up="#26a65b",     ao_dn="#e04040",
    vol="#3a6ea8",       vol_ma="#efb030",
    grid="#1a1e28",      zero="#2a2e3a",
)

plt.rcParams.update({
    "figure.facecolor": STYLE["bg"],
    "axes.facecolor":   STYLE["panel"],
    "axes.edgecolor":   STYLE["border"],
    "axes.labelcolor":  STYLE["muted"],
    "xtick.color":      "#ffffff",
    "ytick.color":      "#ffffff",
    "text.color":       STYLE["text"],
    "grid.color":       STYLE["grid"],
    "grid.linewidth":   0.5,
    "font.family":      "monospace",
    "font.size":        9,
})


# ══════════════════════════════════════════════════════════════════════════════
# TICKERS, GRUPOS E INTERVALOS
# ══════════════════════════════════════════════════════════════════════════════

ALL_TICKERS = [
    "AAPL","MSFT","AMZN","NVDA","GOOG","META","BRK-B","TSLA","JNJ","V",
    "PG","XOM","UNH","JPM","HD","LLY","MA","CVX","ABBV","KO","PEP",
    "COST","BAC","CRM","NFLX","ABT","MCD","LMT","EL","NEE","CAT","MRK",
    "TPL","ASML","ADBE","AVGO","CSCO","CMCSA","AMD","TXN","QCOM","AMAT","LITE","LRCX",
    "INTU","VRTX","ZS","PLTR","CSU.TO","MU","LVMUY","SAP","OR.PA","TTE","SATS","ON",
    "MC.PA","SIE.DE","ENGI.PA","AIR.PA","ALV.DE","EL.PA","AI.PA","BNP.PA",
    "SAN.PA","KER.PA","SU.PA","NESN.SW","LIN.DE","VOW3.DE","BMW.DE","ADS.DE",
    "IFX.DE","MUV2.DE","FRE.DE","DTE.DE","RWE.DE","ITX.MC","BBVA.MC","SAN.MC",
    "TEF.MC","IBE.MC","REP.MC","FER.MC","ACX.MC","ACS.MC","AENA.MC","ANA.MC",
    "IAG.MC","LOG.MC","MAP.MC","PUIG.MC","NTGY.MC","ELE.MC","IDR.MC","PDD",
    "NIO","TCEHY","BZUN","FUTU","MOMO","MNSO","TAL","EDU","WB","XPEV",
    "GC=F","SI=F","BTC-USD","ETH-USD","XRP-USD",
]

GRUPOS = {
    "Todos":             ALL_TICKERS,
    "US Large Cap":      ["AAPL","MSFT","AMZN","NVDA","GOOG","META","BRK-B","TSLA",
                          "JNJ","V","PG","XOM","UNH","JPM","HD","LLY","MA","CVX","ABBV",
                          "KO","PEP","COST","BAC","CRM","NFLX","ABT","MCD","LMT","EL",
                          "NEE","CAT","MRK"],
    "Tecnología":        ["AAPL","MSFT","NVDA","GOOG","META","TSLA","ADBE","AVGO","CSCO",
                          "AMD","TXN","QCOM","AMAT","LRCX","INTU","VRTX","ZS","PLTR","MU",
                          "LITE","ON","ASML","SAP","SIE.DE","IFX.DE","AI.PA"],
    "Europa":            ["MC.PA","SIE.DE","ENGI.PA","AIR.PA","ALV.DE","EL.PA","AI.PA",
                          "BNP.PA","SAN.PA","KER.PA","SU.PA","NESN.SW","LIN.DE","VOW3.DE",
                          "BMW.DE","ADS.DE","IFX.DE","MUV2.DE","FRE.DE","DTE.DE","RWE.DE",
                          "OR.PA","TTE"],
    "España":            ["ITX.MC","BBVA.MC","SAN.MC","TEF.MC","IBE.MC","REP.MC","FER.MC",
                          "ACX.MC","ACS.MC","AENA.MC","ANA.MC","IAG.MC","LOG.MC","MAP.MC",
                          "PUIG.MC","NTGY.MC","ELE.MC","IDR.MC"],
    "China / Asia":      ["PDD","NIO","TCEHY","BZUN","FUTU","MOMO","MNSO","TAL","EDU",
                          "WB","XPEV"],
    "Crypto / Materias": ["GC=F","SI=F","BTC-USD","ETH-USD","XRP-USD"],
}

INTERVAL_CONFIG = {
    "1D": {"yf_interval":"1d",  "yf_period":"2y",  "resample":None, "label":"Diario"},
    "1W": {"yf_interval":"1wk", "yf_period":"5y",  "resample":None, "label":"Semanal"},
    "4h": {"yf_interval":"1h",  "yf_period":"60d", "resample":"4h", "label":"4 horas"},
    "1h": {"yf_interval":"1h",  "yf_period":"30d", "resample":None, "label":"1 hora"},
}


# ══════════════════════════════════════════════════════════════════════════════
# DESCARGA CON INTERVALOS
# ══════════════════════════════════════════════════════════════════════════════

import yfinance as yf

def download_ohlcv(ticker: str, interval_key: str = "1D") -> pd.DataFrame:
    cfg = INTERVAL_CONFIG[interval_key]
    df  = yf.download(
        ticker, period=cfg["yf_period"], interval=cfg["yf_interval"],
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    df = clean_yf_df(df)
    if df.empty:
        return df
    if cfg["resample"]:
        df = df.resample(cfg["resample"]).agg({
            "Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum",
        }).dropna()
    return df


def detectar_divisa(ticker: str) -> str:
    t = ticker.upper()
    if any(t.endswith(s) for s in [".MC", ".PA", ".DE", ".AS", ".BR"]):
        return "EUR"
    elif t.endswith(".SW"):
        return "CHF"
    elif t.endswith(".TO"):
        return "CAD"
    elif t.endswith(".L"):
        return "GBP"
    else:
        return "USD"


# ══════════════════════════════════════════════════════════════════════════════
# CACHÉ
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def cached_dashboard_v2(tickers_tuple: tuple) -> pd.DataFrame:
    return get_sovereign_dashboard_v2(list(tickers_tuple))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_chart_data(ticker: str, interval_key: str = "1D") -> dict:
    df = download_ohlcv(ticker, interval_key)
    if df.empty or len(df) < 60:
        return {}

    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    mcg25     = mcginley_dynamic(close, 25)
    ema200    = EMAIndicator(close=close, window=200).ema_indicator()
    adx_ind   = ADXIndicator(high=high, low=low, close=close, window=14)
    adx_s     = adx_ind.adx()
    pdi_s     = adx_ind.adx_pos()
    ndi_s     = adx_ind.adx_neg()
    ao_s      = awesome_osc(high, low)
    bitman    = clasificar_bitman(df)
    div_df    = detectar_divergencia_simple(df)
    _, bbwp_s = calculate_bbwp(close, bb_len=13, lookback=252)
    konc      = compute_blai5_koncorde(df, m=15)
    pvi_s     = calculate_pvi(close, volume)
    pvi_ema   = pvi_s.ewm(span=25, adjust=False).mean()
    vol_ma    = volume.rolling(20).mean()
    macd_obj  = MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    rsi_s     = RSIIndicator(close=close, window=14).rsi()

    return dict(
        df=df, mcg25=mcg25, ema200=ema200,
        adx_s=adx_s, pdi_s=pdi_s, ndi_s=ndi_s,
        ao_s=ao_s, bitman=bitman, div_df=div_df,
        bbwp_s=bbwp_s, konc=konc,
        pvi_s=pvi_s, pvi_ema=pvi_ema, vol_ma=vol_ma,
        macd_line=macd_obj.macd(),
        macd_sig=macd_obj.macd_signal(),
        macd_hist=macd_obj.macd_diff(),
        rsi_s=rsi_s,
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_precio_actual(ticker: str) -> float:
    """Precio actual para Mi Cartera — caché corto de 5 min."""
    try:
        df = yf.download(ticker, period="2d", interval="1d",
                         auto_adjust=True, progress=False,
                         multi_level_index=False)
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE GRÁFICO
# ══════════════════════════════════════════════════════════════════════════════

def sv(series: pd.Series, index) -> np.ndarray:
    return series.reindex(index).values

def format_xaxis(ax, index, n_labels: int = 8):
    step   = max(1, len(index) // n_labels)
    ticks  = list(range(0, len(index), step))
    labels = [index[i].strftime("%d %b") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8, color="#ffffff")

def panel_style(ax, ylabel: str = "", yticks: int = 5, zero_line: bool = False):
    ax.set_ylabel(ylabel, fontsize=8, labelpad=4, color="#ffffff")
    ax.yaxis.set_major_locator(plt.MaxNLocator(yticks, prune="both"))
    ax.tick_params(axis="both", labelsize=7, colors="#ffffff")
    ax.grid(True, axis="y", linewidth=0.4)
    ax.grid(True, axis="x", linewidth=0.2, alpha=0.4)
    if zero_line:
        ax.axhline(0, color=STYLE["zero"], linewidth=0.8, zorder=1)
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["border"])


# ══════════════════════════════════════════════════════════════════════════════
# SEÑALES RESUMEN (Tab 2 — sin cambios)
# ══════════════════════════════════════════════════════════════════════════════

def build_signals(data: dict) -> list:
    df      = data["df"]
    konc    = data["konc"]
    close   = df["Close"]
    mcg25   = data["mcg25"]
    ema200  = data["ema200"]
    rsi_s   = data["rsi_s"]
    pvi_s   = data["pvi_s"]
    pvi_ema = data["pvi_ema"]
    adx_s   = data["adx_s"]
    bitman  = data["bitman"]
    div_df  = data["div_df"]
    macd_l  = data["macd_line"]
    macd_h  = data["macd_hist"]
    p       = close.iloc[-1]

    def sig(label, bull, neutral=False):
        return {"label": label,
                "state": "neutral" if neutral else ("bull" if bull else "bear")}

    sigs = [
        sig(f"precio {'>' if p >= mcg25.iloc[-1] else '<'} MCG25",   p >= mcg25.iloc[-1]),
        sig(f"precio {'>' if p >= ema200.iloc[-1] else '<'} EMA200", p >= ema200.iloc[-1]),
    ]
    r = rsi_s.iloc[-1]
    sigs.append(sig(f"RSI {r:.1f}", r > 50, neutral=45 < r < 55))
    sigs.append(sig(f"MACD hist {'↑' if macd_h.iloc[-1] >= 0 else '↓'}",
                    macd_h.iloc[-1] >= 0))
    sigs.append(sig(f"MACD línea {'≥0' if macd_l.iloc[-1] >= 0 else '<0'}",
                    macd_l.iloc[-1] >= 0))
    if not konc.empty:
        sigs.append(sig(f"Azul {'↑' if konc['azul'].iloc[-1] >= 0 else '↓'}",
                        konc["azul"].iloc[-1] >= 0))
        sigs.append(sig(f"Verde {'>' if konc['verde'].iloc[-1] >= konc['marron'].iloc[-1] else '<'} M",
                        konc["verde"].iloc[-1] >= konc["marron"].iloc[-1]))
    sigs.append(sig(f"PVI {'>' if pvi_s.iloc[-1] >= pvi_ema.iloc[-1] else '<'} EMA25",
                    pvi_s.iloc[-1] >= pvi_ema.iloc[-1]))
    a = adx_s.iloc[-1]
    sigs.append(sig(f"ADX {a:.1f}", a > 25, neutral=18 < a < 25))
    if bitman is not None and not bitman.empty:
        b_etiq = bitman["Bitman_Etiqueta"].iloc[-1]
        b_v    = int(bitman["Bitman_Velas"].iloc[-1])
        sigs.append(sig(f"Bitman {b_etiq[:8]} ({b_v}v)",
                        b_etiq in ("IMPULSO ALCISTA","RETROCESO ALCISTA"),
                        neutral=(b_etiq == "INDEFINICIÓN")))
    if div_df is not None:
        dt = div_df["divergencia_tipo"].iloc[-1]
        if dt == "alcista":   sigs.append({"label":"Div RSI alc","state":"bull"})
        elif dt == "bajista": sigs.append({"label":"Div RSI baj","state":"bear"})
    return sigs


def score_signals(sigs: list) -> tuple:
    bulls = sum(1 for s in sigs if s["state"] == "bull")
    bears = sum(1 for s in sigs if s["state"] == "bear")
    total = bulls + bears
    pct   = round(bulls / total * 100) if total else 0
    if pct >= 80:   label = "CONFLUENCIA MÁXIMA"
    elif pct >= 60: label = "SETUP SÓLIDO"
    elif pct >= 40: label = "SEÑALES MIXTAS"
    else:           label = "PRESIÓN BAJISTA"
    return pct, label, bulls, len(sigs)


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO MULTI-PANEL (Tab 2 — intacto)
# ══════════════════════════════════════════════════════════════════════════════

def build_figure(data: dict, ticker: str, n_candles: int = 252) -> plt.Figure:
    df        = data["df"]
    close     = df["Close"]
    high      = df["High"]
    low       = df["Low"]
    volume    = df["Volume"]
    mcg25     = data["mcg25"]
    ema200    = data["ema200"]
    adx_s     = data["adx_s"]
    pdi_s     = data["pdi_s"]
    ndi_s     = data["ndi_s"]
    ao_s      = data["ao_s"]
    bitman    = data["bitman"]
    div_df    = data["div_df"]
    bbwp_s    = data["bbwp_s"]
    konc      = data["konc"]
    pvi_s     = data["pvi_s"]
    pvi_ema   = data["pvi_ema"]
    vol_ma    = data["vol_ma"]
    macd_line = data["macd_line"]
    macd_sig  = data["macd_sig"]
    macd_hist = data["macd_hist"]
    rsi_s     = data["rsi_s"]

    sigs                              = build_signals(data)
    pct, score_label, bull_n, total_n = score_signals(sigs)
    score_color = (STYLE["bull"] if pct >= 60
                   else (STYLE["bear"] if pct < 40 else STYLE["mcg"]))

    n_max   = min(n_candles, len(df))
    df_plot = df.iloc[-n_max:]
    idx     = df_plot.index
    xs      = np.arange(len(idx))

    div_alc_xs, div_baj_xs = [], []
    if div_df is not None:
        div_tipos_full = div_df["divergencia_tipo"].reindex(df.index).fillna("ninguna")
        for xi, dt in enumerate(div_tipos_full.iloc[-n_max:]):
            if dt == "alcista":   div_alc_xs.append(xi)
            elif dt == "bajista": div_baj_xs.append(xi)

    fig = plt.figure(figsize=(16, 24), facecolor=STYLE["bg"])
    heights = [5, 1.2, 1.6, 2, 2.2, 1.4, 1.6, 1.8]
    gs = gridspec.GridSpec(8, 1, figure=fig, height_ratios=heights, hspace=0.35)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.04)
    axes = [fig.add_subplot(gs[i]) for i in range(8)]
    for i in range(1, 8):
        axes[i].tick_params(labelbottom=False)

    last_p  = close.iloc[-1]
    prev_p  = close.iloc[-2]
    chg     = last_p - prev_p
    pct_chg = chg / prev_p * 100
    chg_c   = STYLE["bull"] if chg >= 0 else STYLE["bear"]
    sign    = "+" if chg >= 0 else ""
    fig.text(0.07, 0.965, ticker, fontsize=18, fontweight="bold",
             color=STYLE["text"], va="bottom")
    fig.text(0.20, 0.965, f"{last_p:.2f}", fontsize=16, fontweight="bold",
             color=STYLE["text"], va="bottom")
    fig.text(0.32, 0.965, f"{sign}{chg:.2f}  ({sign}{pct_chg:.2f}%)",
             fontsize=12, color=chg_c, va="bottom")
    fig.text(0.97, 0.965, f"{score_label}  ·  {bull_n}/{total_n}  ({pct}%)",
             fontsize=11, color=score_color, ha="right", va="bottom", style="italic")

    # PANEL 0 — Velas
    ax0 = axes[0]; w = 0.4
    for i, (_, row) in enumerate(df_plot.iterrows()):
        col = STYLE["bull"] if row["Close"] >= row["Open"] else STYLE["bear"]
        ax0.plot([i, i], [row["Low"], row["High"]], color=col, lw=0.8, zorder=2)
        ax0.add_patch(plt.Rectangle(
            (i - w, min(row["Open"], row["Close"])),
            2 * w, max(abs(row["Close"] - row["Open"]), 0.001), color=col, zorder=3))
    ax0.plot(xs, sv(mcg25,  idx), color=STYLE["mcg"],   lw=1.4, label="MCG 25",  zorder=4)
    ax0.plot(xs, sv(ema200, idx), color=STYLE["ema200"], lw=1.4, label="EMA 200", zorder=4)
    low_arr  = sv(low,  idx)
    high_arr = sv(high, idx)
    for xi in div_alc_xs:
        if xi < len(low_arr) and not np.isnan(low_arr[xi]):
            ax0.annotate("▲ DIV ALC", xy=(xi, low_arr[xi]),
                         xytext=(0, -14), textcoords="offset points",
                         fontsize=7, color=STYLE["bull"], ha="center", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d3320",
                                   edgecolor=STYLE["bull"], linewidth=0.8))
    for xi in div_baj_xs:
        if xi < len(high_arr) and not np.isnan(high_arr[xi]):
            ax0.annotate("▼ DIV BAJ", xy=(xi, high_arr[xi]),
                         xytext=(0, 14), textcoords="offset points",
                         fontsize=7, color=STYLE["bear"], ha="center", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="#3d0000",
                                   edgecolor=STYLE["bear"], linewidth=0.8))
    ax0.set_xlim(-1, len(idx))
    ax0.legend(loc="upper left", fontsize=8, frameon=False,
               labelcolor=[STYLE["mcg"], STYLE["ema200"]])
    panel_style(ax0, ylabel="Precio")
    ax0.set_title("Velas  ·  McGinley 25  ·  EMA 200",
                  fontsize=9, color=STYLE["muted"], loc="right", pad=4)
    format_xaxis(ax0, idx)
    ax0.tick_params(labelbottom=True, bottom=True)
    ax0.spines["bottom"].set_linewidth(1.2)
    ax0.spines["bottom"].set_edgecolor("#ffffff")

    # PANEL 1 — Volumen
    ax1   = axes[1]
    vol_v = sv(volume, idx)
    vol_m = sv(vol_ma, idx)
    vol_colors = [STYLE["bull"] if df_plot["Close"].iloc[i] >= df_plot["Open"].iloc[i]
                  else STYLE["bear"] for i in range(len(df_plot))]
    ax1.bar(xs, vol_v, color=vol_colors, alpha=0.6, width=0.8, zorder=2)
    ax1.fill_between(xs, vol_m, alpha=0.25, color=STYLE["vol_ma"], zorder=1)
    ax1.plot(xs, vol_m, color=STYLE["vol_ma"], lw=1.2, label="Vol MA20", zorder=3)
    ax1.legend(loc="upper left", fontsize=7, frameon=False, labelcolor=[STYLE["vol_ma"]])
    panel_style(ax1, ylabel="Vol")
    ax1.set_title("Volumen  ·  MA 20", fontsize=9, color=STYLE["muted"], loc="right", pad=4)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))

    # PANEL 2 — RSI
    ax2   = axes[2]
    rsi_v = sv(rsi_s, idx)
    ax2.fill_between(xs, rsi_v, 70, where=(rsi_v > 70), alpha=0.25, color=STYLE["bull"])
    ax2.fill_between(xs, rsi_v, 30, where=(rsi_v < 30), alpha=0.25, color=STYLE["bear"])
    ax2.plot(xs, rsi_v, color=STYLE["rsi"], lw=1.4)
    for lvl, col, ls in [(70,STYLE["bear"],"--"),(50,STYLE["muted"],":"),(30,STYLE["bull"],"--")]:
        ax2.axhline(lvl, color=col, lw=0.7, ls=ls)
    for xi in div_alc_xs:
        if xi < len(rsi_v) and not np.isnan(rsi_v[xi]):
            ax2.annotate("▲", xy=(xi, rsi_v[xi]), fontsize=9, color=STYLE["bull"],
                         ha="center", va="top", xytext=(0,-10), textcoords="offset points",
                         fontweight="bold")
            ax2.axvline(xi, color=STYLE["bull"], lw=0.6, ls=":", alpha=0.5)
    for xi in div_baj_xs:
        if xi < len(rsi_v) and not np.isnan(rsi_v[xi]):
            ax2.annotate("▼", xy=(xi, rsi_v[xi]), fontsize=9, color=STYLE["bear"],
                         ha="center", va="bottom", xytext=(0,10), textcoords="offset points",
                         fontweight="bold")
            ax2.axvline(xi, color=STYLE["bear"], lw=0.6, ls=":", alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_ticks([30, 50, 70])
    panel_style(ax2, ylabel="RSI", yticks=3)
    ax2.set_title("RSI 14  ·  ▲ div alcista  ▼ div bajista",
                  fontsize=9, color=STYLE["muted"], loc="right", pad=4)

    # PANEL 3 — ADX + AO
    ax3  = axes[3]; ax3r = ax3.twinx()
    ao_v = sv(ao_s, idx)
    ao_p = np.roll(ao_v, 1); ao_p[0] = ao_v[0]
    ao_c = [STYLE["ao_up"] if ao_v[i] >= ao_p[i] else STYLE["ao_dn"] for i in range(len(ao_v))]
    ax3r.bar(xs, ao_v, color=ao_c, alpha=0.7, width=0.8, zorder=2)
    ax3r.axhline(0, color=STYLE["zero"], lw=0.7)
    ax3r.tick_params(labelsize=7, colors="#ffffff")
    ax3r.set_ylabel("AO", fontsize=7, color="#ffffff")
    ax3r.spines["right"].set_edgecolor(STYLE["border"])
    ax3.plot(xs, sv(adx_s, idx), color=STYLE["adx"], lw=1.6, label="ADX",  zorder=3)
    ax3.plot(xs, sv(pdi_s, idx), color=STYLE["pdi"], lw=0.9, ls="--", label="+DI", zorder=3)
    ax3.plot(xs, sv(ndi_s, idx), color=STYLE["ndi"], lw=0.9, ls="--", label="-DI", zorder=3)
    ax3.axhline(25, color=STYLE["muted"], lw=0.6, ls=":")
    ax3.legend(loc="upper left", fontsize=7, frameon=False,
               labelcolor=[STYLE["adx"], STYLE["pdi"], STYLE["ndi"]])
    panel_style(ax3, ylabel="ADX")
    ax3.set_title("ADX  ·  +DI / -DI  ·  Awesome Oscillator",
                  fontsize=9, color=STYLE["muted"], loc="right", pad=4)

    # PANEL 4 — Koncorde
    ax4 = axes[4]
    if not konc.empty:
        for key, col in [("verde",STYLE["verde"]),("marron",STYLE["marron"]),("azul",STYLE["azul"])]:
            v = sv(konc[key], idx)
            ax4.fill_between(xs, v, alpha=0.60, color=col, label=key.capitalize(), zorder=2)
            ax4.plot(xs, v, color=col, lw=1.0, zorder=3)
        ax4.plot(xs, sv(konc["media"], idx), color=STYLE["media_k"],
                 lw=1.6, label="Media", zorder=4)
        ax4.legend(loc="upper left", fontsize=7, frameon=False,
                   labelcolor=[STYLE["verde"],STYLE["marron"],STYLE["azul"],STYLE["media_k"]])
    ax4.axhline(0, color=STYLE["zero"], lw=0.7)
    panel_style(ax4, ylabel="Koncorde")
    ax4.set_title("Blai5 Koncorde  ·  Verde / Marrón / Azul / Media",
                  fontsize=9, color=STYLE["muted"], loc="right", pad=4)

    # PANEL 5 — BBWP
    ax5    = axes[5]
    bbwp_v = sv(bbwp_s, idx)
    ax5.fill_between(xs, bbwp_v, 20, where=(~np.isnan(bbwp_v))&(bbwp_v<20),
                     alpha=0.25, color=STYLE["azul"], zorder=1)
    ax5.fill_between(xs, bbwp_v, 80, where=(~np.isnan(bbwp_v))&(bbwp_v>80),
                     alpha=0.25, color=STYLE["bear"], zorder=1)
    for i in range(1, len(xs)):
        if np.isnan(bbwp_v[i]) or np.isnan(bbwp_v[i-1]): continue
        mid = (bbwp_v[i]+bbwp_v[i-1])/2
        lc  = STYLE["azul"] if mid < 20 else (STYLE["bear"] if mid > 80 else STYLE["muted"])
        ax5.plot([xs[i-1],xs[i]], [bbwp_v[i-1],bbwp_v[i]], color=lc, lw=1.5, zorder=3)
    ax5.axhline(80, color=STYLE["bear"],  lw=0.7, ls="--", alpha=0.6)
    ax5.axhline(20, color=STYLE["azul"],  lw=0.7, ls="--", alpha=0.6)
    ax5.axhline(50, color=STYLE["muted"], lw=0.5, ls=":",  alpha=0.4)
    ax5.set_ylim(-2, 102)
    ax5.yaxis.set_ticks([0, 20, 50, 80, 100])
    panel_style(ax5, ylabel="BBWP")
    ax5.set_title("BBWP 13/252  ·  🟢 compresión < 20  ·  🔴 expansión > 80",
                  fontsize=9, color=STYLE["muted"], loc="right", pad=4)

    # PANEL 6 — PVI
    ax6   = axes[6]
    pvi_v = sv(pvi_s,   idx)
    pvi_e = sv(pvi_ema, idx)
    ax6.fill_between(xs, pvi_v, pvi_e, where=(pvi_v>=pvi_e), alpha=0.18, color=STYLE["bull"])
    ax6.fill_between(xs, pvi_v, pvi_e, where=(pvi_v< pvi_e), alpha=0.18, color=STYLE["bear"])
    ax6.plot(xs, pvi_v, color=STYLE["pvi"],     lw=1.4, label="PVI")
    ax6.plot(xs, pvi_e, color=STYLE["pvi_ema"], lw=1.4, ls="--", label="EMA 25")
    ax6.legend(loc="upper left", fontsize=7, frameon=False,
               labelcolor=[STYLE["pvi"], STYLE["pvi_ema"]])
    panel_style(ax6, ylabel="PVI")
    ax6.set_title("PVI  ·  EMA 25", fontsize=9, color=STYLE["muted"], loc="right", pad=4)

    # PANEL 7 — MACD
    ax7    = axes[7]
    hist_v = sv(macd_hist, idx)
    hist_p = np.roll(hist_v, 1); hist_p[0] = hist_v[0]
    bar_col = []
    for i in range(len(hist_v)):
        v, p_v = hist_v[i], hist_p[i]
        if np.isnan(v): bar_col.append(STYLE["muted"]); continue
        bar_col.append(STYLE["bull"] if (v>=0 and v>=p_v) else
                       STYLE["bull_fade"] if v>=0 else
                       STYLE["bear"] if v<=p_v else STYLE["bear_fade"])
    ax7.bar(xs, hist_v, color=bar_col, width=0.8, alpha=0.9, zorder=2)
    ax7.plot(xs, sv(macd_line, idx), color=STYLE["macd_line"], lw=1.3, label="MACD", zorder=3)
    ax7.plot(xs, sv(macd_sig,  idx), color=STYLE["macd_sig"],  lw=1.3, ls="--", label="Señal", zorder=3)
    ax7.axhline(0, color=STYLE["zero"], lw=0.7)
    ax7.legend(loc="upper left", fontsize=7, frameon=False,
               labelcolor=[STYLE["macd_line"], STYLE["macd_sig"]])
    panel_style(ax7, ylabel="MACD")
    format_xaxis(ax7, idx)
    ax7.tick_params(labelbottom=True)
    ax7.set_title("MACD  12 / 26 / 9", fontsize=9, color=STYLE["muted"], loc="right", pad=4)

    for ax in axes:
        ax.set_xlim(-1, len(idx))

    x0  = 0.07
    gap = (0.97 - x0) / len(sigs)
    for i, s in enumerate(sigs):
        col = (STYLE["bull"] if s["state"] == "bull" else
               STYLE["bear"] if s["state"] == "bear" else STYLE["muted"])
        fig.text(x0 + i*gap + (gap-0.003)/2, 0.012, s["label"],
                 fontsize=7.5, color=col, ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=STYLE["panel"],
                           edgecolor=col+"66", linewidth=0.8))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ESTILO DE TABLA
# ══════════════════════════════════════════════════════════════════════════════

SEÑAL_COLOR_V2 = {
    "🚀 POSITIVO CON MOMENTUM": "#26a65b",
    "✅ POSITIVO":               "#4caf50",
    "⚠️ ATENCIÓN KONKORDE":     "#efb030",
    "👀 EN DESARROLLO":          "#6090e0",
    "⏰ POSITIVO MADURO":        "#888888",
    "👀 VIGILAR":                "#555a6a",
    "⛔ SIN SETUP":              "#333333",
}

SALIDA_COLOR = {
    "🟢 MANTENER":             "#26a65b",
    "🟡 VIGILAR POSICIÓN":     "#efb030",
    "🟠 CONSIDERAR REDUCIR":   "#e08030",
    "🔴 SALIDA":               "#e04040",
}


def color_señal_v2(val: str) -> str:
    col = SEÑAL_COLOR_V2.get(val, "#ffffff")
    return f"color: {col}; font-weight: bold"


def color_salida(val: str) -> str:
    col = SALIDA_COLOR.get(val, "#ffffff")
    return f"color: {col}; font-weight: bold"


def style_df_v2(df: pd.DataFrame):
    styler = df.style
    fn = styler.map if hasattr(styler, "map") else styler.applymap
    return (
        fn(color_señal_v2, subset=["Señal"])
        .set_properties(**{
            "background-color": "#13161e",
            "color":            "#ffffff",
            "border-color":     "#1f2430",
            "font-size":        "0.80rem",
        })
        .set_table_styles([{"selector":"th","props":[
            ("background-color","#0d0f14"),("color","#efb030"),
            ("font-size","0.82rem"),("border-bottom","1px solid #1f2430"),
        ]}])
    )


# ══════════════════════════════════════════════════════════════════════════════
# HISTÓRICO DE SEÑALES
# ══════════════════════════════════════════════════════════════════════════════

def guardar_historico(df_nuevo: pd.DataFrame):
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    if "hist_prev" not in st.session_state:
        st.session_state["hist_prev"]      = None
        st.session_state["hist_prev_hora"] = None
    st.session_state["hist_prev"]        = st.session_state.get("hist_actual", None)
    st.session_state["hist_prev_hora"]   = st.session_state.get("hist_actual_hora", None)
    st.session_state["hist_actual"]      = df_nuevo.copy()
    st.session_state["hist_actual_hora"] = now


def mostrar_cambios():
    prev      = st.session_state.get("hist_prev", None)
    curr      = st.session_state.get("hist_actual", None)
    hora_prev = st.session_state.get("hist_prev_hora", "—")
    hora_curr = st.session_state.get("hist_actual_hora", "—")
    if prev is None or curr is None:
        st.info("Necesitas al menos dos ejecuciones para ver cambios.")
        return
    merged  = curr[["Ticker","Señal"]].merge(
        prev[["Ticker","Señal"]].rename(columns={"Señal":"Señal_Ant"}),
        on="Ticker", how="left")
    cambios = merged[merged["Señal"] != merged["Señal_Ant"]].dropna()
    st.caption(f"Comparando: **{hora_curr}** (actual) vs **{hora_prev}** (anterior)")
    if cambios.empty:
        st.success("✅ Sin cambios de señal entre los dos últimos análisis.")
        return
    st.markdown(f"**{len(cambios)} tickers cambiaron de señal:**")
    for _, row in cambios.iterrows():
        col_ant = SEÑAL_COLOR_V2.get(row["Señal_Ant"], "#888")
        col_new = SEÑAL_COLOR_V2.get(row["Señal"],     "#888")
        st.markdown(
            f"**{row['Ticker']}** &nbsp;&nbsp;"
            f"<span style='color:{col_ant}'>{row['Señal_Ant']}</span>"
            f" &nbsp;→&nbsp; "
            f"<span style='color:{col_new}'>{row['Señal']}</span>",
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MI CARTERA — persistencia CSV
# ══════════════════════════════════════════════════════════════════════════════

CARTERA_FILE = "cartera.csv"
HISTORIAL_FILE = "historial_operaciones.csv"

CARTERA_COLS   = ["Ticker","Divisa","Precio_Entrada","Fecha_Entrada","Capital","Notas"]
HISTORIAL_COLS = ["Ticker","Divisa","Precio_Entrada","Precio_Salida",
                  "Fecha_Entrada","Fecha_Salida","Capital","PnL_pct","PnL_divisa","Notas"]


def cargar_cartera() -> pd.DataFrame:
    if os.path.exists(CARTERA_FILE):
        try:
            return pd.read_csv(CARTERA_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=CARTERA_COLS)


def guardar_cartera(df: pd.DataFrame):
    df.to_csv(CARTERA_FILE, index=False)


def cargar_historial() -> pd.DataFrame:
    if os.path.exists(HISTORIAL_FILE):
        try:
            return pd.read_csv(HISTORIAL_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=HISTORIAL_COLS)


def guardar_historial(df: pd.DataFrame):
    df.to_csv(HISTORIAL_FILE, index=False)


def añadir_posicion(ticker, divisa, precio_entrada, fecha_entrada, capital, notas):
    df = cargar_cartera()
    nueva = pd.DataFrame([{
        "Ticker":         ticker,
        "Divisa":         divisa,
        "Precio_Entrada": precio_entrada,
        "Fecha_Entrada":  fecha_entrada,
        "Capital":        capital,
        "Notas":          notas,
    }])
    df = pd.concat([df, nueva], ignore_index=True)
    guardar_cartera(df)


def cerrar_posicion(idx: int, precio_salida: float):
    df_cart = cargar_cartera()
    df_hist = cargar_historial()
    row     = df_cart.iloc[idx].copy()

    pnl_pct    = (precio_salida - row["Precio_Entrada"]) / row["Precio_Entrada"] * 100
    pnl_divisa = row["Capital"] * pnl_pct / 100

    nueva_hist = pd.DataFrame([{
        "Ticker":         row["Ticker"],
        "Divisa":         row["Divisa"],
        "Precio_Entrada": row["Precio_Entrada"],
        "Precio_Salida":  precio_salida,
        "Fecha_Entrada":  row["Fecha_Entrada"],
        "Fecha_Salida":   datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "Capital":        row["Capital"],
        "PnL_pct":        round(pnl_pct, 2),
        "PnL_divisa":     round(pnl_divisa, 2),
        "Notas":          row["Notas"],
    }])

    df_hist = pd.concat([df_hist, nueva_hist], ignore_index=True)
    guardar_historial(df_hist)

    df_cart = df_cart.drop(index=idx).reset_index(drop=True)
    guardar_cartera(df_cart)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Semáforo")
    st.markdown("---")
    grupo_sel = st.selectbox(
        "Grupo de tickers", options=list(GRUPOS.keys()), index=0, key="grupo")
    custom_input = st.text_area(
        "Tickers personalizados\n(uno por línea)", height=120, key="custom_tickers")
    force_refresh_tab1 = st.button("🔄 Recalcular semáforo", key="refresh1")
    st.markdown("---")
    st.markdown(
        "<small style='color:#aaaaaa'>Datos: Yahoo Finance · Caché 1h<br>"
        "BBWP config 13/252 estructural</small>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑAS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📊  DASHBOARD SEÑALES",
    "📈  GRÁFICOS ESTRATEGIA",
    "💼  MI CARTERA",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Dashboard Señales (nuevo scoring v2)
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("## 📊 Dashboard Señales — Sovereign v3")
    st.caption(
        "🚀 POSITIVO CON MOMENTUM = ≥6/8 activas y ≥4 frescas  ·  "
        "Frescura = señal ocurrió en las últimas 3 velas  ·  "
        "🔥 = señal fresca"
    )

    if custom_input.strip():
        tickers_tab1 = [t.strip().upper()
                        for t in custom_input.strip().splitlines() if t.strip()]
    else:
        tickers_tab1 = GRUPOS[grupo_sel]

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Tickers seleccionados", len(tickers_tab1))

    cache_key = tuple(sorted(tickers_tab1))
    if force_refresh_tab1:
        cached_dashboard_v2.clear()

    with st.spinner(f"Calculando {len(tickers_tab1)} tickers…"):
        pb  = st.progress(0)
        stx = st.empty()
        df_result = cached_dashboard_v2(cache_key)
        pb.progress(100)
        stx.empty()

    now_str = datetime.now(timezone.utc).strftime("%d/%m/%Y  %H:%M UTC")
    st.caption(f"🕐 Última actualización: **{now_str}**")

    if df_result.empty:
        st.warning("No se obtuvieron resultados. Revisa la conexión o los tickers.")
    else:
        guardar_historico(df_result)

        momentum_count = len(df_result[df_result["Señal"] == "🚀 POSITIVO CON MOMENTUM"])
        positivo_count = len(df_result[df_result["Señal"].isin(
            ["🚀 POSITIVO CON MOMENTUM", "✅ POSITIVO", "⚠️ ATENCIÓN KONKORDE"])])
        col_m2.metric("Procesados", len(df_result))
        col_m3.metric("🚀 Con Momentum", momentum_count)
        col_m4.metric("✅ Positivos total", positivo_count)

        filtro_señal = st.multiselect(
            "Filtrar por señal",
            options=list(df_result["Señal"].unique()),
            default=[],
            key="filtro_señal",
        )
        df_show = (df_result[df_result["Señal"].isin(filtro_señal)]
                   if filtro_señal else df_result)

        cols_tabla = [c for c in df_show.columns if c != "Detalle"]
        st.dataframe(
            style_df_v2(df_show[cols_tabla]),
            use_container_width=True,
            height=min(600, 38 + 35 * len(df_show)),
        )

        with st.expander("🔍 Ver detalle señales"):
            df_det = df_show[["Ticker","Señal","Activas","Frescas","Detalle"]].copy()
            _s  = df_det.style
            _fn = _s.map if hasattr(_s, "map") else _s.applymap
            st.dataframe(
                _fn(color_señal_v2, subset=["Señal"])
                .set_properties(**{
                    "background-color":"#13161e","color":"#ffffff",
                    "font-size":"0.72rem","white-space":"nowrap",
                }),
                use_container_width=True, height=400,
            )

        with st.expander("📊 Cambios vs análisis anterior"):
            mostrar_cambios()

        st.markdown("""
---
**Leyenda señales v2**

| Señal | Condición |
|---|---|
| 🚀 POSITIVO CON MOMENTUM | ≥6/8 activas y ≥4 frescas (≤3 velas) |
| ✅ POSITIVO | ≥5/8 activas y ≥2 frescas |
| ⏰ POSITIVO MADURO | ≥5/8 activas pero <2 frescas — señal tardía |
| ⚠️ ATENCIÓN KONKORDE | Azul K positivo y Verde K negativo |
| 👀 EN DESARROLLO | 3-4/8 activas con ≥2 frescas |
| 👀 VIGILAR | 3-4/8 activas sin frescura |
| ⛔ SIN SETUP | <3 activas |

**Las 8 señales:**  S1 MACD cruce↑  ·  S2 AO rojo→verde  ·  S3 PVI cruza EMA25↑  ·  
S4 Azul K cruza 0↑  ·  S5 Media K en área  ·  S6 Bitman impulso↑  ·  
S7 BBWP pendiente↑  ·  S8 Volumen confirm.  ·  🔥 = fresca (≤3v)
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Gráficos (intacto)
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("## 📈 Gráficos Estrategia")

    col_a, col_b, col_c, col_d = st.columns([5, 1, 1, 1])
    with col_a:
        if "chart_tickers_sel" not in st.session_state:
            st.session_state["chart_tickers_sel"] = ["NVDA"]
        chart_tickers = st.multiselect(
            "📌 Tickers para graficar",
            options=sorted(ALL_TICKERS),
            default=st.session_state["chart_tickers_sel"],
            key="chart_tickers",
        )
        st.session_state["chart_tickers_sel"] = chart_tickers
    with col_b:
        chart_interval = st.selectbox(
            "Intervalo", options=list(INTERVAL_CONFIG.keys()), index=0, key="chart_interval")
    with col_c:
        zoom_candles = st.selectbox(
            "Zoom", options=[50,100,150,252,365,500], index=3, key="zoom_candles")
    with col_d:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        force_refresh_tab2 = st.button("🔄 Recalcular", key="refresh2")

    if chart_interval in ("1h","4h"):
        st.warning(f"⚠️ Intervalo **{chart_interval}**: indicadores de largo plazo orientativos.")

    st.caption("8 paneles: Velas · Volumen · RSI · ADX+AO · Koncorde · BBWP · PVI · MACD")
    st.markdown("---")

    if not chart_tickers:
        st.info("👆 Selecciona al menos un ticker.")
        st.stop()

    if force_refresh_tab2:
        cached_chart_data.clear()

    for chart_ticker in chart_tickers:
        st.markdown(
            f"### 📊 {chart_ticker}  "
            f"<small style='color:#aaaaaa'>· {INTERVAL_CONFIG[chart_interval]['label']}"
            f" · {zoom_candles} velas</small>", unsafe_allow_html=True)

        with st.spinner(f"Calculando {chart_ticker}…"):
            chart_data = cached_chart_data(chart_ticker, interval_key=chart_interval)

        if not chart_data:
            st.error(f"Sin datos para **{chart_ticker}**.")
            continue

        now_str2 = datetime.now(timezone.utc).strftime("%d/%m/%Y  %H:%M UTC")
        st.caption(f"🕐 Última actualización: **{now_str2}**")

        df_c   = chart_data["df"]
        close  = df_c["Close"]
        m1,m2,m3,m4,m5 = st.columns(5)
        last_p = close.iloc[-1]; prev_p = close.iloc[-2]
        chg = last_p - prev_p;   pct_c = chg / prev_p * 100
        m1.metric("Último precio", f"{last_p:.2f}", f"{chg:+.2f} ({pct_c:+.2f}%)")
        rsi_val = chart_data["rsi_s"].iloc[-1]
        m2.metric("RSI 14", f"{rsi_val:.1f}",
                  "Sobrecompra" if rsi_val > 70 else ("Sobreventa" if rsi_val < 30 else "Neutral"))
        bbwp_v = chart_data["bbwp_s"].dropna()
        bbwp_l = bbwp_v.iloc[-1] if len(bbwp_v) > 0 else np.nan
        m3.metric("BBWP 13/252",
                  f"{bbwp_l:.1f}%" if not np.isnan(bbwp_l) else "n/d",
                  "compresión" if bbwp_l < 20 else ("expansión" if bbwp_l > 80 else "normal"))
        mcg_v = chart_data["mcg25"].iloc[-1]
        m4.metric("McGinley 25", f"{mcg_v:.2f}",
                  "↑ sobre MCG" if last_p > mcg_v else "↓ bajo MCG")
        e200_v = chart_data["ema200"].iloc[-1]
        m5.metric("EMA 200", f"{e200_v:.2f}",
                  "↑ sobre EMA" if last_p > e200_v else "↓ bajo EMA")

        with st.spinner(f"Renderizando {chart_ticker}…"):
            fig = build_figure(chart_data, chart_ticker, n_candles=zoom_candles)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("#### Señales actuales")
        sigs = build_signals(chart_data)
        pct_s, label_s, bull_n, total_n = score_signals(sigs)
        cols_sig = st.columns(len(sigs))
        for col, s in zip(cols_sig, sigs):
            color = "🟢" if s["state"]=="bull" else ("🔴" if s["state"]=="bear" else "⚪")
            col.markdown(
                f"<div style='text-align:center;font-size:0.7rem;color:#ffffff;'>"
                f"{color}<br>{s['label']}</div>", unsafe_allow_html=True)
        score_col = "green" if pct_s >= 60 else ("red" if pct_s < 40 else "orange")
        st.markdown(
            f"<h4 style='color:{score_col};text-align:center;'>"
            f"{label_s}  ·  {bull_n}/{total_n}  ({pct_s}%)</h4>",
            unsafe_allow_html=True)

        buf = io.BytesIO()
        fig_dl = build_figure(chart_data, chart_ticker, n_candles=zoom_candles)
        fig_dl.savefig(buf, format="png", dpi=150,
                       bbox_inches="tight", facecolor=STYLE["bg"])
        plt.close(fig_dl)
        buf.seek(0)
        st.download_button(
            label=f"⬇️ Descargar PNG — {chart_ticker}", data=buf,
            file_name=f"sovereign_{chart_ticker.replace('=','').replace('-','_')}.png",
            mime="image/png", key=f"dl_{chart_ticker}_{chart_interval}")
        st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Mi Cartera
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown("## 💼 Mi Cartera")
    st.caption("Seguimiento de posiciones abiertas con semáforo de salida en tiempo real.")

    # ── Formulario añadir posición ────────────────────────────────────────
    with st.expander("➕ Añadir nueva posición", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            new_ticker = st.selectbox(
                "Ticker", options=sorted(ALL_TICKERS), key="new_ticker")
            new_divisa = st.text_input(
                "Divisa (autodetectada)",
                value=detectar_divisa(new_ticker),
                key="new_divisa",
            )
        with fc2:
            new_precio = st.number_input(
                "Precio de entrada", min_value=0.0, value=0.0,
                step=0.01, format="%.4f", key="new_precio")
            new_capital = st.number_input(
                "Capital invertido", min_value=0.0, value=1000.0,
                step=100.0, key="new_capital")
        with fc3:
            new_fecha = st.date_input(
                "Fecha de entrada",
                value=datetime.now(timezone.utc).date(),
                key="new_fecha")
            new_notas = st.text_input(
                "Notas (opcional)", key="new_notas")

        if st.button("✅ Añadir posición", key="btn_añadir"):
            if new_ticker and new_precio > 0:
                añadir_posicion(
                    ticker=new_ticker,
                    divisa=new_divisa,
                    precio_entrada=new_precio,
                    fecha_entrada=str(new_fecha),
                    capital=new_capital,
                    notas=new_notas,
                )
                st.success(f"✅ {new_ticker} añadido a la cartera.")
                st.rerun()
            else:
                st.error("Introduce un ticker y un precio de entrada válido.")

    # ── Cargar cartera ────────────────────────────────────────────────────
    df_cartera = cargar_cartera()

    if df_cartera.empty:
        st.info("No tienes posiciones abiertas. Añade una posición arriba. 👆")
    else:
        st.markdown(f"### Posiciones abiertas ({len(df_cartera)})")
        st.caption(f"🕐 Precios actualizados: {datetime.now(timezone.utc).strftime('%H:%M UTC')}")

        # ── Calcular precios actuales y semáforos ─────────────────────────
        rows_display = []

        for idx, row in df_cartera.iterrows():
            ticker        = row["Ticker"]
            precio_entrada = float(row["Precio_Entrada"])
            capital       = float(row["Capital"])
            divisa        = row["Divisa"]
            fecha_entrada = row["Fecha_Entrada"]

            # precio actual
            precio_actual = get_precio_actual(ticker)
            if precio_actual <= 0:
                precio_actual = precio_entrada

            # P&L
            pnl_pct    = (precio_actual - precio_entrada) / precio_entrada * 100
            pnl_divisa = capital * pnl_pct / 100

            # días en cartera
            try:
                dias = (datetime.now(timezone.utc).date() -
                        pd.to_datetime(fecha_entrada).date()).days
            except Exception:
                dias = "—"

            # semáforo de salida
            with st.spinner(f"Calculando señales salida {ticker}…"):
                chart_d = cached_chart_data(ticker, interval_key="1D")

            if chart_d:
                sal = semaforo_salida(
                    df=chart_d["df"],
                    kdf=chart_d["konc"],
                    bitman_df=chart_d["bitman"],
                    macd_line=chart_d["macd_line"],
                    macd_sig=chart_d["macd_sig"],
                    pvi_s=chart_d["pvi_s"],
                    pvi_ema=chart_d["pvi_ema"],
                )
                etiq_salida   = sal["etiqueta"]
                razones_salida = sal["razones"]
            else:
                etiq_salida    = "⚪ Sin datos"
                razones_salida = ""

            rows_display.append({
                "idx":           idx,
                "Ticker":        ticker,
                "Divisa":        divisa,
                "Entrada":       f"{precio_entrada:.4f}",
                "Actual":        f"{precio_actual:.4f}",
                "P&L %":         f"{pnl_pct:+.2f}%",
                "P&L":           f"{pnl_divisa:+.2f} {divisa}",
                "Días":          dias,
                "Fecha entrada": fecha_entrada,
                "Semáforo":      etiq_salida,
                "Razones salida": razones_salida,
                "Notas":         row.get("Notas",""),
            })

        # ── Tabla resumen ──────────────────────────────────────────────────
        df_display = pd.DataFrame(rows_display)
        cols_tabla = ["Ticker","Divisa","Entrada","Actual","P&L %","P&L",
                      "Días","Fecha entrada","Semáforo"]

        styler = df_display[cols_tabla].style
        fn     = styler.map if hasattr(styler, "map") else styler.applymap
        st.dataframe(
            fn(color_salida, subset=["Semáforo"])
            .set_properties(**{
                "background-color": "#13161e",
                "color":            "#ffffff",
                "font-size":        "0.82rem",
            })
            .set_table_styles([{"selector":"th","props":[
                ("background-color","#0d0f14"),("color","#efb030"),
                ("font-size","0.84rem"),("border-bottom","1px solid #1f2430"),
            ]}]),
            use_container_width=True,
            height=min(500, 38 + 40 * len(df_display)),
        )

        # ── Detalle por posición ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 Detalle por posición")

        for row_d in rows_display:
            ticker    = row_d["Ticker"]
            etiq      = row_d["Semáforo"]
            col_etiq  = SALIDA_COLOR.get(etiq, "#ffffff")
            pnl_color = "green" if "+" in row_d["P&L %"] else "red"

            with st.expander(
                f"{'🔴' if 'SALIDA' in etiq else '🟠' if 'REDUCIR' in etiq else '🟡' if 'VIGILAR' in etiq else '🟢'}"
                f"  {ticker}  —  "
                f"P&L: {row_d['P&L %']}  ·  {etiq}"
            ):
                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Entrada",  row_d["Entrada"])
                dc2.metric("Actual",   row_d["Actual"])
                dc3.metric("P&L %",    row_d["P&L %"])
                dc4.metric("P&L",      row_d["P&L"])

                st.markdown(
                    f"**Semáforo salida:** "
                    f"<span style='color:{col_etiq};font-weight:bold'>{etiq}</span>",
                    unsafe_allow_html=True)
                st.caption(f"Señales: {row_d['Razones salida']}")

                if row_d["Notas"]:
                    st.caption(f"📝 {row_d['Notas']}")

                # cerrar posición
                st.markdown("**Cerrar posición:**")
                cc1, cc2 = st.columns([2, 1])
                with cc1:
                    precio_cierre = st.number_input(
                        "Precio de salida",
                        min_value=0.0,
                        value=float(row_d["Actual"].replace(",",".")),
                        step=0.01,
                        format="%.4f",
                        key=f"cierre_{ticker}_{row_d['idx']}",
                    )
                with cc2:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                    if st.button(f"🔒 Cerrar {ticker}",
                                 key=f"btn_cerrar_{ticker}_{row_d['idx']}"):
                        cerrar_posicion(row_d["idx"], precio_cierre)
                        st.success(f"✅ {ticker} cerrado. Guardado en historial.")
                        st.rerun()

        # ── Resumen de cartera ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Resumen")
        usd_rows = [r for r in rows_display if r["Divisa"] == "USD"]
        eur_rows = [r for r in rows_display if r["Divisa"] == "EUR"]

        def calcular_pnl_total(rows):
            total = sum(float(r["P&L"].split()[0].replace(",",".")) for r in rows)
            return total

        rs1, rs2, rs3 = st.columns(3)
        rs1.metric("Posiciones abiertas", len(rows_display))
        if usd_rows:
            rs2.metric("P&L total USD",
                       f"{calcular_pnl_total(usd_rows):+.2f} $")
        if eur_rows:
            rs3.metric("P&L total EUR",
                       f"{calcular_pnl_total(eur_rows):+.2f} €")

    # ── Historial de operaciones cerradas ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Historial de operaciones cerradas")
    df_hist = cargar_historial()

    if df_hist.empty:
        st.info("Aún no has cerrado ninguna posición.")
    else:
        def color_pnl(val):
            try:
                v = float(str(val).replace("%","").replace(",","."))
                return f"color: {'#26a65b' if v >= 0 else '#e04040'}; font-weight: bold"
            except Exception:
                return ""

        styler_h = df_hist.style
        fn_h     = styler_h.map if hasattr(styler_h, "map") else styler_h.applymap
        st.dataframe(
            fn_h(color_pnl, subset=["PnL_pct","PnL_divisa"])
            .set_properties(**{
                "background-color":"#13161e","color":"#ffffff","font-size":"0.80rem"})
            .set_table_styles([{"selector":"th","props":[
                ("background-color","#0d0f14"),("color","#efb030"),("font-size","0.82rem")]}]),
            use_container_width=True,
        )

        # resumen historial
        if "PnL_pct" in df_hist.columns:
            ganadas  = len(df_hist[df_hist["PnL_pct"] > 0])
            perdidas = len(df_hist[df_hist["PnL_pct"] <= 0])
            media    = df_hist["PnL_pct"].mean()
            rh1, rh2, rh3, rh4 = st.columns(4)
            rh1.metric("Operaciones cerradas", len(df_hist))
            rh2.metric("Ganadoras", ganadas)
            rh3.metric("Perdedoras", perdidas)
            rh4.metric("P&L medio", f"{media:+.2f}%")
