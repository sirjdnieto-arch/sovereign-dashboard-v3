# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Sovereign Dashboard v3  —  Streamlit  —  3 pestañas
#   Tab 1 : Dashboard Señales  (scoring v2 + Sniper V6.2 INDEPENDIENTE)
#   Tab 2 : Gráficos Estrategia  (panel Sniper añadido)
#   Tab 3 : Mi Cartera  (columna Sniper)
#
# NOTA: Motor LCrack V6.2 Sniper embebido directamente en este archivo.
#       NO se importa lcrack_sniper.py — evita ImportError en Streamlit Cloud.
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
    calcular_señales_v2,
    semaforo_salida,
    get_sovereign_dashboard_v2,
)

import yfinance as yf


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR LCRACK V6.2 SNIPER — embebido (sin archivo externo)
# ══════════════════════════════════════════════════════════════════════════════

def calcular_motor_v62_sniper(df: pd.DataFrame) -> pd.DataFrame:
    """
    Motor matemático LCrack V6.2 Sniper.
    Parámetros calibrados (Techo Matemático V6.2):
      - Ventana macro (inercia):   120 velas
      - Ventana gatillo (corto):    10 velas
      - Umbral inercia ULTRA:     0.00045 (slope normalizado)
      - Umbral inercia MONITOR:   0.00010
      - Rango BBWP válido:        [0.15, 0.40]
      - Zona precio:              precio <= reg_largo + 0.10 * ATR

    Columnas añadidas al df:
      Sniper_State            — "ULTRA-SAFE SNIPER" | "MONITOR" | "IGNORAR"
      Sniper_Detalle          — razón textual
      Sniper_Velas_Activacion — velas desde último setup válido
    """
    if df is None or df.empty or len(df) < 120:
        if df is not None:
            df["Sniper_State"]            = "IGNORAR"
            df["Sniper_Detalle"]          = "Historial insuficiente (< 120 velas)"
            df["Sniper_Velas_Activacion"] = 999
        return df

    close_vals = df["Close"].values
    w_long     = 120
    w_short    = 10

    # 1. ATR Estructural (14) ─────────────────────────────────────────────────
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"]  - df["Close"].shift(1)).abs()
    df["ATR_Sniper"] = (
        pd.concat([tr1, tr2, tr3], axis=1)
        .max(axis=1)
        .rolling(window=14)
        .mean()
    )

    # 2. BBWP Institucional (13/252) ─────────────────────────────────────────
    _mid = df["Close"].rolling(window=13).mean().replace(0, np.nan)
    _std = df["Close"].rolling(window=13).std()
    _bbw = (_std * 4) / _mid
    df["BBWP_Sniper"] = _bbw.rolling(window=252).apply(
        lambda x: float((x[-1] >= x).sum()) / len(x) if len(x) > 0 else 0.0,
        raw=True,
    )

    # 3. Regresiones Lineales Normalizadas ───────────────────────────────────
    slopes_long   = np.zeros(len(df))
    reg_long_vals = np.zeros(len(df))
    slopes_short  = np.zeros(len(df))

    for i in range(w_long, len(df)):
        sl = close_vals[: i + 1]

        # Macro (120)
        y_l   = sl[-w_long:]
        m_l, b_l = np.polyfit(np.arange(w_long), y_l, 1)
        mean_l   = np.mean(y_l)
        slopes_long[i]   = m_l / mean_l if mean_l != 0 else 0.0
        reg_long_vals[i] = m_l * (w_long - 1) + b_l

        # Gatillo (10)
        y_s    = sl[-w_short:]
        m_s, _ = np.polyfit(np.arange(w_short), y_s, 1)
        mean_s = np.mean(y_s)
        slopes_short[i]  = m_s / mean_s if mean_s != 0 else 0.0

    df["slope_long_sn"]       = slopes_long
    df["reg_long_sn"]         = reg_long_vals
    df["slope_short_sn"]      = slopes_short
    df["slope_short_prev_sn"] = df["slope_short_sn"].shift(1)

    # 4. Clasificar cada vela ────────────────────────────────────────────────
    states           = []
    razones_lista    = []
    velas_act        = []
    ticks_since_setup = 999

    for i in range(len(df)):
        if i < w_long:
            states.append("IGNORAR")
            razones_lista.append("Historial insuficiente")
            velas_act.append(999)
            continue

        precio   = df["Close"].iloc[i]
        reg_l    = df["reg_long_sn"].iloc[i]
        atr      = df["ATR_Sniper"].iloc[i]
        s_long   = df["slope_long_sn"].iloc[i]
        s_short  = df["slope_short_sn"].iloc[i]
        s_sp     = df["slope_short_prev_sn"].iloc[i]
        bbwp_val = df["BBWP_Sniper"].iloc[i]

        # ── Setup: zona de valor + inercia ──────────────────────────────
        if (s_long > 0.00045) and (precio <= reg_l + 0.10 * atr):
            ticks_since_setup = 0
        else:
            ticks_since_setup += 1

        cond_trigger = (s_short > 0) and (s_sp <= 0)
        cond_bbwp    = 0.15 <= bbwp_val <= 0.40

        # ── ULTRA-SAFE SNIPER ───────────────────────────────────────────
        if cond_trigger and (ticks_since_setup <= 1) and cond_bbwp:
            states.append("ULTRA-SAFE SNIPER")
            razones_lista.append(
                f"Sniper V6.2 | Inercia:{s_long:.5f} BBWP:{bbwp_val:.2f}"
            )
            velas_act.append(0)
            continue

        # ── MONITOR ─────────────────────────────────────────────────────
        if s_long > 0.00010 and precio <= reg_l:
            states.append("MONITOR")
            razones_lista.append(
                f"Inercia alcista ({s_long:.5f}). Esperando gatillo en suelo."
            )
        else:
            states.append("IGNORAR")
            if s_long <= 0:
                razones_lista.append("Sin inercia estructural (bajista/lateral).")
            elif bbwp_val < 0.15:
                razones_lista.append(f"Volatilidad muerta (BBWP:{bbwp_val:.2f}).")
            else:
                razones_lista.append("Precio extendido fuera de parámetros.")

        velas_act.append(ticks_since_setup + 1)

    df["Sniper_State"]            = states
    df["Sniper_Detalle"]          = razones_lista
    df["Sniper_Velas_Activacion"] = velas_act
    return df


def get_sniper_status(df: pd.DataFrame) -> dict:
    """Wrapper — devuelve estado Sniper de la ÚLTIMA vela del df."""
    try:
        df_s = calcular_motor_v62_sniper(df.copy())
        return {
            "state": df_s["Sniper_State"].iloc[-1],
            "razon": df_s["Sniper_Detalle"].iloc[-1],
            "velas": int(df_s["Sniper_Velas_Activacion"].iloc[-1]),
        }
    except Exception as e:
        return {"state": "IGNORAR", "razon": f"Error: {str(e)[:60]}", "velas": 999}


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
    button[data-baseweb="tab"][aria-selected="true"]::after { display: none !important; }
    @media (max-width: 768px) {
        .stDataFrame tbody td { font-size: 0.65rem !important; }
        h2 { font-size: 1.1rem !important; }
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
    sniper_ultra="#00e5ff",
    sniper_monitor="#efb030",
    sniper_ignorar="#555a6a",
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
# TICKERS Y GRUPOS
# ══════════════════════════════════════════════════════════════════════════════

ALL_TICKERS = [
    "AAPL","MSFT","AMZN","NVDA","GOOG","META","BRK-B","TSLA","JNJ","V",
    "PG","XOM","UNH","JPM","HD","LLY","MA","CVX","ABBV","KO","PEP",
    "COST","BAC","CRM","NFLX","ABT","MCD","LMT","EL","NEE","CAT","MRK",
    "TPL","ASML","ADBE","AVGO","CSCO","CMCSA","AMD","TXN","QCOM","AMAT","LITE","LRCX","COHR","CMI",
    "NEM","ULTA","IT","FOXA","LUV","VLO","ADP","FN","POET","KEYS","HPE","MRVL","BRKR","AAOI",
    "INTU","VRTX","ZS","PLTR","CSU.TO","MU","LVMUY","SAP","OR.PA","TTE","SATS","ON","MELI","CTSH","THRY","KLTR","QBTS","RGTI","IONQ",
    "MC.PA","SIE.DE","ENGI.PA","AIR.PA","ALV.DE","EL.PA","AI.PA","BNP.PA",
    "SAN.PA","KER.PA","SU.PA","NESN.SW","LIN.DE","VOW3.DE","BMW.DE","ADS.DE",
    "IFX.DE","MUV2.DE","FRE.DE","DTE.DE","RWE.DE","ITX.MC","BBVA.MC","SAN.MC",
    "TEF.MC","IBE.MC","REP.MC","FER.MC","ACX.MC","ACS.MC","AENA.MC","ANA.MC",
    "IAG.MC","LOG.MC","MAP.MC","PUIG.MC","NTGY.MC","ELE.MC","IDR.MC","PDD",
    "NIO","TCEHY","BZUN","FUTU","MOMO","MNSO","TAL","EDU","WB","XPEV",
    "GC=F","SI=F","BTC-USD","ETH-USD","XRP-USD","SOL-USD","CRCL"
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
    "China / Asia":      ["PDD","NIO","TCEHY","BZUN","FUTU","MOMO","MNSO","TAL","EDU","WB","XPEV"],
    "Crypto / Materias": ["GC=F","SI=F","BTC-USD","ETH-USD","XRP-USD"],
}

INTERVAL_CONFIG = {
    "1D": {"yf_interval":"1d",  "yf_period":"2y",  "resample":None, "label":"Diario"},
    "1W": {"yf_interval":"1wk", "yf_period":"5y",  "resample":None, "label":"Semanal"},
    "4h": {"yf_interval":"1h",  "yf_period":"60d", "resample":"4h", "label":"4 horas"},
    "1h": {"yf_interval":"1h",  "yf_period":"30d", "resample":None, "label":"1 hora"},
}


# ══════════════════════════════════════════════════════════════════════════════
# DESCARGA OHLCV
# ══════════════════════════════════════════════════════════════════════════════

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
    if any(t.endswith(s) for s in [".MC",".PA",".DE",".AS",".BR"]): return "EUR"
    elif t.endswith(".SW"):  return "CHF"
    elif t.endswith(".TO"):  return "CAD"
    elif t.endswith(".L"):   return "GBP"
    return "USD"


# ══════════════════════════════════════════════════════════════════════════════
# CACHÉ
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def cached_dashboard_v2(tickers_tuple: tuple) -> pd.DataFrame:
    """
    Llama a get_sovereign_dashboard_v2 y añade columnas Sniper independientes.
    El scoring Pepino v2 NO se ve afectado.
    """
    df_result = get_sovereign_dashboard_v2(list(tickers_tuple))
    if df_result.empty:
        return df_result

    sniper_states, sniper_razones, sniper_velas_list = [], [], []

    for ticker in df_result["Ticker"]:
        try:
            raw = yf.download(
                ticker, period="2y", interval="1d",
                auto_adjust=True, progress=False, multi_level_index=False,
            )
            raw = clean_yf_df(raw)
            if raw.empty or len(raw) < 120:
                raise ValueError("Datos insuficientes")
            st_info = get_sniper_status(raw)
        except Exception as e:
            st_info = {"state":"IGNORAR", "razon":f"Error:{str(e)[:40]}", "velas":999}

        sniper_states.append(st_info["state"])
        sniper_razones.append(st_info["razon"])
        sniper_velas_list.append(st_info["velas"])

    df_result["Sniper"]       = sniper_states
    df_result["Sniper_Razón"] = sniper_razones
    df_result["Sniper_Velas"] = sniper_velas_list
    return df_result


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

    # ── Sniper (independiente, con try/except) ────────────────────────────
    sniper_state, sniper_razon, sniper_velas = "IGNORAR", "", 999
    try:
        st_info      = get_sniper_status(df.copy())
        sniper_state = st_info["state"]
        sniper_razon = st_info["razon"]
        sniper_velas = st_info["velas"]
    except Exception as e:
        sniper_razon = f"Error Sniper: {str(e)[:50]}"

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
        sniper_state=sniper_state,
        sniper_razon=sniper_razon,
        sniper_velas=sniper_velas,
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_precio_actual(ticker: str) -> float:
    try:
        df = yf.download(ticker, period="2d", interval="1d",
                         auto_adjust=True, progress=False, multi_level_index=False)
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS GRÁFICO
# ══════════════════════════════════════════════════════════════════════════════

def sv(series: pd.Series, index) -> np.ndarray:
    return series.reindex(index).values

def format_xaxis(ax, index, n_labels: int = 8):
    step   = max(1, len(index) // n_labels)
    ticks  = list(range(0, len(index), step))
    labels = [index[i].strftime("%d %b") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8, color="#ffffff")

def panel_style(ax, ylabel="", yticks=5, zero_line=False):
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
# SEÑALES RESUMEN
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
        return {"label":label, "state":"neutral" if neutral else ("bull" if bull else "bear")}

    sigs = [
        sig(f"precio {'>' if p >= mcg25.iloc[-1] else '<'} MCG25",   p >= mcg25.iloc[-1]),
        sig(f"precio {'>' if p >= ema200.iloc[-1] else '<'} EMA200", p >= ema200.iloc[-1]),
    ]
    r = rsi_s.iloc[-1]
    sigs.append(sig(f"RSI {r:.1f}", r > 50, neutral=45 < r < 55))
    sigs.append(sig(f"MACD hist {'↑' if macd_h.iloc[-1] >= 0 else '↓'}", macd_h.iloc[-1] >= 0))
    sigs.append(sig(f"MACD línea {'≥0' if macd_l.iloc[-1] >= 0 else '<0'}", macd_l.iloc[-1] >= 0))
    if not konc.empty:
        sigs.append(sig(f"Azul {'↑' if konc['azul'].iloc[-1] >= 0 else '↓'}", konc["azul"].iloc[-1] >= 0))
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
# HELPER UI: badge Sniper
# ══════════════════════════════════════════════════════════════════════════════

def sniper_badge_html(state: str, velas: int = None) -> str:
    if state == "ULTRA-SAFE SNIPER":
        color = STYLE["sniper_ultra"]
        label = f"🔥 ULTRA-SAFE SNIPER{f'  ({velas}v)' if velas is not None else ''}"
    elif state == "MONITOR":
        color = STYLE["sniper_monitor"]
        label = f"👁️ MONITOR{f'  ({velas}v)' if velas is not None else ''}"
    else:
        color = STYLE["sniper_ignorar"]
        label = "❌ IGNORAR"
    return (f"<span style='color:{color};font-weight:bold;"
            f"background:{color}22;padding:2px 10px;border-radius:99px;"
            f"border:1px solid {color}55;'>{label}</span>")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO MULTI-PANEL
# ══════════════════════════════════════════════════════════════════════════════

def build_figure(data: dict, ticker: str, n_candles: int = 252) -> plt.Figure:
    df        = data["df"]
    close     = df["Close"]; high = df["High"]; low = df["Low"]; volume = df["Volume"]
    mcg25     = data["mcg25"]; ema200 = data["ema200"]
    adx_s     = data["adx_s"]; pdi_s = data["pdi_s"]; ndi_s = data["ndi_s"]
    ao_s      = data["ao_s"]; bitman = data["bitman"]; div_df = data["div_df"]
    bbwp_s    = data["bbwp_s"]; konc = data["konc"]
    pvi_s     = data["pvi_s"]; pvi_ema = data["pvi_ema"]; vol_ma = data["vol_ma"]
    macd_line = data["macd_line"]; macd_sig = data["macd_sig"]; macd_hist = data["macd_hist"]
    rsi_s     = data["rsi_s"]
    sniper_state = data.get("sniper_state", "IGNORAR")
    sniper_velas = data.get("sniper_velas", 999)

    sigs                              = build_signals(data)
    pct, score_label, bull_n, total_n = score_signals(sigs)
    score_color = STYLE["bull"] if pct >= 60 else (STYLE["bear"] if pct < 40 else STYLE["mcg"])

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
    gs  = gridspec.GridSpec(8, 1, figure=fig, height_ratios=heights, hspace=0.35)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.04)
    axes = [fig.add_subplot(gs[i]) for i in range(8)]
    for i in range(1, 8): axes[i].tick_params(labelbottom=False)

    last_p  = close.iloc[-1]; prev_p = close.iloc[-2]
    chg     = last_p - prev_p; pct_chg = chg / prev_p * 100
    chg_c   = STYLE["bull"] if chg >= 0 else STYLE["bear"]
    sign    = "+" if chg >= 0 else ""
    fig.text(0.07, 0.965, ticker,         fontsize=18, fontweight="bold", color=STYLE["text"], va="bottom")
    fig.text(0.20, 0.965, f"{last_p:.2f}",fontsize=16, fontweight="bold", color=STYLE["text"], va="bottom")
    fig.text(0.32, 0.965, f"{sign}{chg:.2f}  ({sign}{pct_chg:.2f}%)", fontsize=12, color=chg_c, va="bottom")
    fig.text(0.97, 0.965, f"{score_label}  ·  {bull_n}/{total_n}  ({pct}%)",
             fontsize=11, color=score_color, ha="right", va="bottom", style="italic")

    # Sniper watermark
    sniper_col = (STYLE["sniper_ultra"] if sniper_state == "ULTRA-SAFE SNIPER"
                  else STYLE["sniper_monitor"] if sniper_state == "MONITOR"
                  else STYLE["sniper_ignorar"])
    sniper_lbl = (f"🔥 SNIPER ({sniper_velas}v)" if sniper_state == "ULTRA-SAFE SNIPER"
                  else "👁 MONITOR" if sniper_state == "MONITOR" else "SNIPER ❌")
    fig.text(0.52, 0.965, sniper_lbl, fontsize=10, color=sniper_col,
             ha="center", va="bottom", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=sniper_col+"22",
                       edgecolor=sniper_col+"88", linewidth=0.8))

    # PANEL 0 — Velas
    ax0 = axes[0]; w = 0.4
    for i, (_, row) in enumerate(df_plot.iterrows()):
        col = STYLE["bull"] if row["Close"] >= row["Open"] else STYLE["bear"]
        ax0.plot([i,i],[row["Low"],row["High"]], color=col, lw=0.8, zorder=2)
        ax0.add_patch(plt.Rectangle((i-w, min(row["Open"],row["Close"])),
            2*w, max(abs(row["Close"]-row["Open"]),0.001), color=col, zorder=3))
    ax0.plot(xs, sv(mcg25,idx),  color=STYLE["mcg"],   lw=1.4, label="MCG 25",  zorder=4)
    ax0.plot(xs, sv(ema200,idx), color=STYLE["ema200"], lw=1.4, label="EMA 200", zorder=4)
    low_arr=sv(low,idx); high_arr=sv(high,idx)
    for xi in div_alc_xs:
        if xi<len(low_arr) and not np.isnan(low_arr[xi]):
            ax0.annotate("▲ DIV ALC", xy=(xi,low_arr[xi]), xytext=(0,-14),
                         textcoords="offset points", fontsize=7, color=STYLE["bull"],
                         ha="center", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.2",facecolor="#0d3320",edgecolor=STYLE["bull"],linewidth=0.8))
    for xi in div_baj_xs:
        if xi<len(high_arr) and not np.isnan(high_arr[xi]):
            ax0.annotate("▼ DIV BAJ", xy=(xi,high_arr[xi]), xytext=(0,14),
                         textcoords="offset points", fontsize=7, color=STYLE["bear"],
                         ha="center", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.2",facecolor="#3d0000",edgecolor=STYLE["bear"],linewidth=0.8))
    ax0.set_xlim(-1,len(idx))
    ax0.legend(loc="upper left",fontsize=8,frameon=False,labelcolor=[STYLE["mcg"],STYLE["ema200"]])
    panel_style(ax0,ylabel="Precio")
    ax0.set_title("Velas  ·  McGinley 25  ·  EMA 200",fontsize=9,color=STYLE["muted"],loc="right",pad=4)
    format_xaxis(ax0,idx)
    ax0.tick_params(labelbottom=True,bottom=True)
    ax0.spines["bottom"].set_linewidth(1.2)
    ax0.spines["bottom"].set_edgecolor("#ffffff")

    # PANEL 1 — Volumen
    ax1=axes[1]
    vol_v=sv(volume,idx); vol_m=sv(vol_ma,idx)
    vol_colors=[STYLE["bull"] if df_plot["Close"].iloc[i]>=df_plot["Open"].iloc[i] else STYLE["bear"] for i in range(len(df_plot))]
    ax1.bar(xs,vol_v,color=vol_colors,alpha=0.6,width=0.8,zorder=2)
    ax1.fill_between(xs,vol_m,alpha=0.25,color=STYLE["vol_ma"],zorder=1)
    ax1.plot(xs,vol_m,color=STYLE["vol_ma"],lw=1.2,label="Vol MA20",zorder=3)
    ax1.legend(loc="upper left",fontsize=7,frameon=False,labelcolor=[STYLE["vol_ma"]])
    panel_style(ax1,ylabel="Vol")
    ax1.set_title("Volumen  ·  MA 20",fontsize=9,color=STYLE["muted"],loc="right",pad=4)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x/1e6:.0f}M" if x>=1e6 else f"{x/1e3:.0f}K"))

    # PANEL 2 — RSI
    ax2=axes[2]; rsi_v=sv(rsi_s,idx)
    ax2.fill_between(xs,rsi_v,70,where=(rsi_v>70),alpha=0.25,color=STYLE["bull"])
    ax2.fill_between(xs,rsi_v,30,where=(rsi_v<30),alpha=0.25,color=STYLE["bear"])
    ax2.plot(xs,rsi_v,color=STYLE["rsi"],lw=1.4)
    for lvl,col,ls in [(70,STYLE["bear"],"--"),(50,STYLE["muted"],":"),(30,STYLE["bull"],"--")]:
        ax2.axhline(lvl,color=col,lw=0.7,ls=ls)
    for xi in div_alc_xs:
        if xi<len(rsi_v) and not np.isnan(rsi_v[xi]):
            ax2.annotate("▲",xy=(xi,rsi_v[xi]),fontsize=9,color=STYLE["bull"],ha="center",va="top",
                         xytext=(0,-10),textcoords="offset points",fontweight="bold")
            ax2.axvline(xi,color=STYLE["bull"],lw=0.6,ls=":",alpha=0.5)
    for xi in div_baj_xs:
        if xi<len(rsi_v) and not np.isnan(rsi_v[xi]):
            ax2.annotate("▼",xy=(xi,rsi_v[xi]),fontsize=9,color=STYLE["bear"],ha="center",va="bottom",
                         xytext=(0,10),textcoords="offset points",fontweight="bold")
            ax2.axvline(xi,color=STYLE["bear"],lw=0.6,ls=":",alpha=0.5)
    ax2.set_ylim(0,100); ax2.yaxis.set_ticks([30,50,70])
    panel_style(ax2,ylabel="RSI",yticks=3)
    ax2.set_title("RSI 14  ·  ▲ div alcista  ▼ div bajista",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 3 — ADX + AO
    ax3=axes[3]; ax3r=ax3.twinx()
    ao_v=sv(ao_s,idx); ao_p=np.roll(ao_v,1); ao_p[0]=ao_v[0]
    ao_c=[STYLE["ao_up"] if ao_v[i]>=ao_p[i] else STYLE["ao_dn"] for i in range(len(ao_v))]
    ax3r.bar(xs,ao_v,color=ao_c,alpha=0.7,width=0.8,zorder=2)
    ax3r.axhline(0,color=STYLE["zero"],lw=0.7)
    ax3r.tick_params(labelsize=7,colors="#ffffff")
    ax3r.set_ylabel("AO",fontsize=7,color="#ffffff")
    ax3r.spines["right"].set_edgecolor(STYLE["border"])
    ax3.plot(xs,sv(adx_s,idx),color=STYLE["adx"],lw=1.6,label="ADX",zorder=3)
    ax3.plot(xs,sv(pdi_s,idx),color=STYLE["pdi"],lw=0.9,ls="--",label="+DI",zorder=3)
    ax3.plot(xs,sv(ndi_s,idx),color=STYLE["ndi"],lw=0.9,ls="--",label="-DI",zorder=3)
    ax3.axhline(25,color=STYLE["muted"],lw=0.6,ls=":")
    ax3.legend(loc="upper left",fontsize=7,frameon=False,labelcolor=[STYLE["adx"],STYLE["pdi"],STYLE["ndi"]])
    panel_style(ax3,ylabel="ADX")
    ax3.set_title("ADX  ·  +DI / -DI  ·  Awesome Oscillator",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 4 — Koncorde
    ax4=axes[4]
    if not konc.empty:
        for key,col in [("verde",STYLE["verde"]),("marron",STYLE["marron"]),("azul",STYLE["azul"])]:
            v=sv(konc[key],idx)
            ax4.fill_between(xs,v,alpha=0.60,color=col,label=key.capitalize(),zorder=2)
            ax4.plot(xs,v,color=col,lw=1.0,zorder=3)
        ax4.plot(xs,sv(konc["media"],idx),color=STYLE["media_k"],lw=1.6,label="Media",zorder=4)
        ax4.legend(loc="upper left",fontsize=7,frameon=False,
                   labelcolor=[STYLE["verde"],STYLE["marron"],STYLE["azul"],STYLE["media_k"]])
    ax4.axhline(0,color=STYLE["zero"],lw=0.7)
    panel_style(ax4,ylabel="Koncorde")
    ax4.set_title("Blai5 Koncorde  ·  Verde / Marrón / Azul / Media",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 5 — BBWP
    ax5=axes[5]; bbwp_v=sv(bbwp_s,idx)
    ax5.fill_between(xs,bbwp_v,20,where=(~np.isnan(bbwp_v))&(bbwp_v<20),alpha=0.25,color=STYLE["azul"],zorder=1)
    ax5.fill_between(xs,bbwp_v,80,where=(~np.isnan(bbwp_v))&(bbwp_v>80),alpha=0.25,color=STYLE["bear"],zorder=1)
    for i in range(1,len(xs)):
        if np.isnan(bbwp_v[i]) or np.isnan(bbwp_v[i-1]): continue
        mid=(bbwp_v[i]+bbwp_v[i-1])/2
        lc=STYLE["azul"] if mid<20 else(STYLE["bear"] if mid>80 else STYLE["muted"])
        ax5.plot([xs[i-1],xs[i]],[bbwp_v[i-1],bbwp_v[i]],color=lc,lw=1.5,zorder=3)
    ax5.axhline(80,color=STYLE["bear"],lw=0.7,ls="--",alpha=0.6)
    ax5.axhline(20,color=STYLE["azul"],lw=0.7,ls="--",alpha=0.6)
    ax5.axhline(50,color=STYLE["muted"],lw=0.5,ls=":",alpha=0.4)
    ax5.set_ylim(-2,102); ax5.yaxis.set_ticks([0,20,50,80,100])
    panel_style(ax5,ylabel="BBWP")
    ax5.set_title("BBWP 13/252  ·  compresión<20  ·  expansión>80",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 6 — PVI
    ax6=axes[6]; pvi_v=sv(pvi_s,idx); pvi_e=sv(pvi_ema,idx)
    ax6.fill_between(xs,pvi_v,pvi_e,where=(pvi_v>=pvi_e),alpha=0.18,color=STYLE["bull"])
    ax6.fill_between(xs,pvi_v,pvi_e,where=(pvi_v<pvi_e), alpha=0.18,color=STYLE["bear"])
    ax6.plot(xs,pvi_v,color=STYLE["pvi"],    lw=1.4,label="PVI")
    ax6.plot(xs,pvi_e,color=STYLE["pvi_ema"],lw=1.4,ls="--",label="EMA 25")
    ax6.legend(loc="upper left",fontsize=7,frameon=False,labelcolor=[STYLE["pvi"],STYLE["pvi_ema"]])
    panel_style(ax6,ylabel="PVI")
    ax6.set_title("PVI  ·  EMA 25",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 7 — MACD
    ax7=axes[7]; hist_v=sv(macd_hist,idx)
    hist_p=np.roll(hist_v,1); hist_p[0]=hist_v[0]
    bar_col=[]
    for i in range(len(hist_v)):
        v,p_v=hist_v[i],hist_p[i]
        if np.isnan(v): bar_col.append(STYLE["muted"]); continue
        bar_col.append(STYLE["bull"] if(v>=0 and v>=p_v) else
                       STYLE["bull_fade"] if v>=0 else
                       STYLE["bear"] if v<=p_v else STYLE["bear_fade"])
    ax7.bar(xs,hist_v,color=bar_col,width=0.8,alpha=0.9,zorder=2)
    ax7.plot(xs,sv(macd_line,idx),color=STYLE["macd_line"],lw=1.3,label="MACD",zorder=3)
    ax7.plot(xs,sv(macd_sig,idx), color=STYLE["macd_sig"], lw=1.3,ls="--",label="Señal",zorder=3)
    ax7.axhline(0,color=STYLE["zero"],lw=0.7)
    ax7.legend(loc="upper left",fontsize=7,frameon=False,labelcolor=[STYLE["macd_line"],STYLE["macd_sig"]])
    panel_style(ax7,ylabel="MACD"); format_xaxis(ax7,idx)
    ax7.tick_params(labelbottom=True)
    ax7.set_title("MACD  12/26/9",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    for ax in axes: ax.set_xlim(-1,len(idx))

    x0=0.07; gap=(0.97-x0)/len(sigs)
    for i,s in enumerate(sigs):
        col=(STYLE["bull"] if s["state"]=="bull" else STYLE["bear"] if s["state"]=="bear" else STYLE["muted"])
        fig.text(x0+i*gap+(gap-0.003)/2, 0.012, s["label"],
                 fontsize=7.5,color=col,ha="center",va="center",
                 bbox=dict(boxstyle="round,pad=0.3",facecolor=STYLE["panel"],edgecolor=col+"66",linewidth=0.8))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ESTILO TABLAS
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
    "🟢 MANTENER":           "#26a65b",
    "🟡 VIGILAR POSICIÓN":   "#efb030",
    "🟠 CONSIDERAR REDUCIR": "#e08030",
    "🔴 SALIDA":             "#e04040",
}
SNIPER_COLOR = {
    "ULTRA-SAFE SNIPER": "#00e5ff",
    "MONITOR":           "#efb030",
    "IGNORAR":           "#555a6a",
}

def color_señal_v2(val): return f"color:{SEÑAL_COLOR_V2.get(val,'#fff')};font-weight:bold"
def color_salida(val):   return f"color:{SALIDA_COLOR.get(val,'#fff')};font-weight:bold"
def color_sniper(val):   return f"color:{SNIPER_COLOR.get(val,'#aaa')};font-weight:bold"

def style_df_v2(df: pd.DataFrame):
    _s = df.style
    fn = _s.map if hasattr(_s,"map") else _s.applymap
    s  = fn(color_señal_v2, subset=["Señal"])
    if "Sniper" in df.columns:
        fn2 = s.map if hasattr(s,"map") else s.applymap
        s   = fn2(color_sniper, subset=["Sniper"])
    return s.set_properties(**{
        "background-color":"#13161e","color":"#ffffff",
        "border-color":"#1f2430","font-size":"0.80rem",
    }).set_table_styles([{"selector":"th","props":[
        ("background-color","#0d0f14"),("color","#efb030"),
        ("font-size","0.82rem"),("border-bottom","1px solid #1f2430"),
    ]}])


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
    prev=st.session_state.get("hist_prev",None)
    curr=st.session_state.get("hist_actual",None)
    hora_prev=st.session_state.get("hist_prev_hora","—")
    hora_curr=st.session_state.get("hist_actual_hora","—")
    if prev is None or curr is None:
        st.info("Necesitas al menos dos ejecuciones para ver cambios.")
        return
    merged=curr[["Ticker","Señal"]].merge(
        prev[["Ticker","Señal"]].rename(columns={"Señal":"Señal_Ant"}),on="Ticker",how="left")
    cambios=merged[merged["Señal"]!=merged["Señal_Ant"]].dropna()
    st.caption(f"Comparando: **{hora_curr}** (actual) vs **{hora_prev}** (anterior)")
    if cambios.empty:
        st.success("✅ Sin cambios de señal entre los dos últimos análisis.")
        return
    st.markdown(f"**{len(cambios)} tickers cambiaron de señal:**")
    for _,row in cambios.iterrows():
        col_ant=SEÑAL_COLOR_V2.get(row["Señal_Ant"],"#888")
        col_new=SEÑAL_COLOR_V2.get(row["Señal"],"#888")
        st.markdown(
            f"**{row['Ticker']}** &nbsp;&nbsp;"
            f"<span style='color:{col_ant}'>{row['Señal_Ant']}</span>"
            f" &nbsp;→&nbsp; <span style='color:{col_new}'>{row['Señal']}</span>",
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CARTERA — persistencia CSV
# ══════════════════════════════════════════════════════════════════════════════

CARTERA_FILE   = "cartera.csv"
HISTORIAL_FILE = "historial_operaciones.csv"
CARTERA_COLS   = ["Ticker","Divisa","Precio_Entrada","Fecha_Entrada","Capital","Notas"]
HISTORIAL_COLS = ["Ticker","Divisa","Precio_Entrada","Precio_Salida",
                  "Fecha_Entrada","Fecha_Salida","Capital","PnL_pct","PnL_divisa","Notas"]

def cargar_cartera() -> pd.DataFrame:
    if os.path.exists(CARTERA_FILE):
        try: return pd.read_csv(CARTERA_FILE)
        except Exception: pass
    return pd.DataFrame(columns=CARTERA_COLS)

def guardar_cartera(df): df.to_csv(CARTERA_FILE, index=False)

def cargar_historial() -> pd.DataFrame:
    if os.path.exists(HISTORIAL_FILE):
        try: return pd.read_csv(HISTORIAL_FILE)
        except Exception: pass
    return pd.DataFrame(columns=HISTORIAL_COLS)

def guardar_historial(df): df.to_csv(HISTORIAL_FILE, index=False)

def añadir_posicion(ticker,divisa,precio_entrada,fecha_entrada,capital,notas):
    df=cargar_cartera()
    nueva=pd.DataFrame([{"Ticker":ticker,"Divisa":divisa,"Precio_Entrada":precio_entrada,
                          "Fecha_Entrada":fecha_entrada,"Capital":capital,"Notas":notas}])
    guardar_cartera(pd.concat([df,nueva],ignore_index=True))

def cerrar_posicion(idx: int, precio_salida: float):
    df_cart=cargar_cartera(); df_hist=cargar_historial()
    row=df_cart.iloc[idx].copy()
    pnl_pct   =(precio_salida-row["Precio_Entrada"])/row["Precio_Entrada"]*100
    pnl_divisa=row["Capital"]*pnl_pct/100
    nueva_hist=pd.DataFrame([{
        "Ticker":row["Ticker"],"Divisa":row["Divisa"],
        "Precio_Entrada":row["Precio_Entrada"],"Precio_Salida":precio_salida,
        "Fecha_Entrada":row["Fecha_Entrada"],
        "Fecha_Salida":datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "Capital":row["Capital"],"PnL_pct":round(pnl_pct,2),
        "PnL_divisa":round(pnl_divisa,2),"Notas":row["Notas"],
    }])
    guardar_historial(pd.concat([df_hist,nueva_hist],ignore_index=True))
    guardar_cartera(df_cart.drop(index=idx).reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Semáforo")
    st.markdown("---")
    grupo_sel    = st.selectbox("Grupo de tickers",options=list(GRUPOS.keys()),index=0,key="grupo")
    custom_input = st.text_area("Tickers personalizados\n(uno por línea)",height=120,key="custom_tickers")
    force_refresh_tab1 = st.button("🔄 Recalcular semáforo",key="refresh1")
    st.markdown("---")
    st.markdown("### 🎯 Sniper V6.2")
    mostrar_sniper_col = st.checkbox("Mostrar columna Sniper",value=True,key="mostrar_sniper")
    solo_sniper        = st.checkbox("Solo tickers con Sniper activo",value=False,key="solo_sniper")
    st.markdown("---")
    st.markdown("<small style='color:#aaa'>Datos: Yahoo Finance · Caché 1h<br>BBWP 13/252 estructural</small>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑAS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📊  DASHBOARD SEÑALES",
    "📈  GRÁFICOS ESTRATEGIA",
    "💼  MI CARTERA",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Dashboard Señales
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("## 📊 Dashboard Señales — Sovereign v3  +  🎯 Sniper V6.2")
    st.caption(
        "🚀 POSITIVO CON MOMENTUM = ≥6/8 activas y ≥4 frescas  ·  "
        "Sniper = indicador INDEPENDIENTE (no afecta scoring Pepino)  ·  🔥 = señal fresca"
    )

    tickers_tab1 = (
        [t.strip().upper() for t in custom_input.strip().splitlines() if t.strip()]
        if custom_input.strip() else GRUPOS[grupo_sel]
    )

    col_m1,col_m2,col_m3,col_m4,col_m5 = st.columns(5)
    col_m1.metric("Tickers seleccionados", len(tickers_tab1))

    cache_key = tuple(sorted(tickers_tab1))
    if force_refresh_tab1:
        cached_dashboard_v2.clear()

    with st.spinner(f"Calculando {len(tickers_tab1)} tickers + Sniper…"):
        pb=st.progress(0); stx=st.empty()
        df_result = cached_dashboard_v2(cache_key)
        pb.progress(100); stx.empty()

    now_str = datetime.now(timezone.utc).strftime("%d/%m/%Y  %H:%M UTC")
    st.caption(f"🕐 Última actualización: **{now_str}**")

    if df_result.empty:
        st.warning("No se obtuvieron resultados.")
    else:
        guardar_historico(df_result)

        momentum_count = len(df_result[df_result["Señal"] == "🚀 POSITIVO CON MOMENTUM"])
        positivo_count = len(df_result[df_result["Señal"].isin(
            ["🚀 POSITIVO CON MOMENTUM","✅ POSITIVO","⚠️ ATENCIÓN KONKORDE"])])
        sniper_count   = len(df_result[df_result.get("Sniper","") == "ULTRA-SAFE SNIPER"]) if "Sniper" in df_result.columns else 0
        col_m2.metric("Procesados",         len(df_result))
        col_m3.metric("🚀 Con Momentum",    momentum_count)
        col_m4.metric("✅ Positivos total", positivo_count)
        col_m5.metric("🔥 Sniper activo",   sniper_count)

        fc1,fc2 = st.columns([2,1])
        with fc1:
            filtro_señal = st.multiselect("Filtrar por señal Pepino",
                options=list(df_result["Señal"].unique()),default=[],key="filtro_señal")
        with fc2:
            filtro_sniper_tab1 = st.multiselect("🎯 Filtrar por Sniper",
                options=["ULTRA-SAFE SNIPER","MONITOR","IGNORAR"],default=[],
                key="filtro_sniper_tab1")

        df_show = df_result.copy()
        if filtro_señal:       df_show = df_show[df_show["Señal"].isin(filtro_señal)]
        if filtro_sniper_tab1: df_show = df_show[df_show["Sniper"].isin(filtro_sniper_tab1)]
        if solo_sniper and "Sniper" in df_show.columns:
            df_show = df_show[df_show["Sniper"] == "ULTRA-SAFE SNIPER"]

        cols_excluir = {"Detalle","Sniper_Razón"}
        if not mostrar_sniper_col: cols_excluir.update({"Sniper","Sniper_Velas"})
        cols_tabla = [c for c in df_show.columns if c not in cols_excluir]
        st.dataframe(style_df_v2(df_show[cols_tabla]),
                     use_container_width=True, height=min(600,38+35*len(df_show)))

        with st.expander("🔍 Ver detalle señales"):
            df_det=df_show[["Ticker","Señal","Activas","Frescas","Detalle"]].copy()
            _s=df_det.style; _fn=_s.map if hasattr(_s,"map") else _s.applymap
            st.dataframe(_fn(color_señal_v2,subset=["Señal"])
                         .set_properties(**{"background-color":"#13161e","color":"#ffffff",
                                            "font-size":"0.72rem","white-space":"nowrap"}),
                         use_container_width=True, height=400)

        with st.expander("🎯 Ver detalle Sniper V6.2"):
            if "Sniper" in df_show.columns:
                df_sn=df_show[["Ticker","Señal","Sniper","Sniper_Velas","Sniper_Razón"]].copy()
                _ss=df_sn.style; _sf=_ss.map if hasattr(_ss,"map") else _ss.applymap
                st.dataframe(_sf(color_sniper,subset=["Sniper"])
                             .set_properties(**{"background-color":"#13161e","color":"#ffffff",
                                                "font-size":"0.72rem"}),
                             use_container_width=True, height=400)

        with st.expander("📊 Cambios vs análisis anterior"):
            mostrar_cambios()

        st.markdown("""
---
**Leyenda señales v2**

| Señal | Condición |
|---|---|
| 🚀 POSITIVO CON MOMENTUM | ≥6/8 activas y ≥4 frescas (≤3 velas) |
| ✅ POSITIVO | ≥5/8 activas y ≥2 frescas |
| ⏰ POSITIVO MADURO | ≥5/8 activas pero <2 frescas |
| ⚠️ ATENCIÓN KONKORDE | Azul K positivo y Verde K negativo |
| 👀 EN DESARROLLO | 3-4/8 activas con ≥2 frescas |
| 👀 VIGILAR | 3-4/8 activas sin frescura |
| ⛔ SIN SETUP | <3 activas |

**Sniper V6.2 (independiente):**
🔥 ULTRA-SAFE SNIPER = inercia macro >0.00045 + precio en zona + gatillo corto + BBWP [0.15-0.40]
👁️ MONITOR = inercia >0.00010 y precio ≤ regresión larga · ❌ IGNORAR = sin condiciones
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Gráficos
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("## 📈 Gráficos Estrategia")

    col_a,col_b,col_c,col_d = st.columns([5,1,1,1])
    with col_a:
        if "chart_tickers_sel" not in st.session_state:
            st.session_state["chart_tickers_sel"] = ["NVDA"]
        chart_tickers = st.multiselect("📌 Tickers para graficar",
            options=sorted(ALL_TICKERS),
            default=st.session_state["chart_tickers_sel"],key="chart_tickers")
        st.session_state["chart_tickers_sel"] = chart_tickers
    with col_b:
        chart_interval = st.selectbox("Intervalo",options=list(INTERVAL_CONFIG.keys()),index=0,key="chart_interval")
    with col_c:
        zoom_candles = st.selectbox("Zoom",options=[50,100,150,252,365,500],index=3,key="zoom_candles")
    with col_d:
        st.markdown("<div style='height:28px'></div>",unsafe_allow_html=True)
        force_refresh_tab2 = st.button("🔄 Recalcular",key="refresh2")

    if chart_interval in ("1h","4h"):
        st.warning(f"⚠️ Intervalo **{chart_interval}**: indicadores de largo plazo orientativos.")
    st.caption("8 paneles: Velas · Volumen · RSI · ADX+AO · Koncorde · BBWP · PVI · MACD")
    st.markdown("---")

    if not chart_tickers:
        st.info("👆 Selecciona al menos un ticker.")
        st.stop()

    if force_refresh_tab2: cached_chart_data.clear()

    for chart_ticker in chart_tickers:
        st.markdown(f"### 📊 {chart_ticker}  "
                    f"<small style='color:#aaa'>· {INTERVAL_CONFIG[chart_interval]['label']} · {zoom_candles} velas</small>",
                    unsafe_allow_html=True)

        with st.spinner(f"Calculando {chart_ticker}…"):
            chart_data = cached_chart_data(chart_ticker, interval_key=chart_interval)

        if not chart_data:
            st.error(f"Sin datos para **{chart_ticker}**."); continue

        st.caption(f"🕐 {datetime.now(timezone.utc).strftime('%d/%m/%Y  %H:%M UTC')}")

        # Banner Sniper
        sniper_state = chart_data.get("sniper_state","IGNORAR")
        sniper_razon = chart_data.get("sniper_razon","")
        sniper_velas = chart_data.get("sniper_velas",999)
        st.markdown(f"🎯 **Sniper V6.2:** {sniper_badge_html(sniper_state,sniper_velas)}",
                    unsafe_allow_html=True)
        st.caption(f"🔍 {sniper_razon}")

        df_c=chart_data["df"]; close=df_c["Close"]
        m1,m2,m3,m4,m5 = st.columns(5)
        last_p=close.iloc[-1]; prev_p=close.iloc[-2]
        chg=last_p-prev_p; pct_c=chg/prev_p*100
        m1.metric("Último precio",f"{last_p:.2f}",f"{chg:+.2f} ({pct_c:+.2f}%)")
        rsi_val=chart_data["rsi_s"].iloc[-1]
        m2.metric("RSI 14",f"{rsi_val:.1f}",
                  "Sobrecompra" if rsi_val>70 else("Sobreventa" if rsi_val<30 else "Neutral"))
        bbwp_v=chart_data["bbwp_s"].dropna()
        bbwp_l=bbwp_v.iloc[-1] if len(bbwp_v)>0 else np.nan
        m3.metric("BBWP 13/252",
                  f"{bbwp_l:.1f}%" if not np.isnan(bbwp_l) else "n/d",
                  "compresión" if bbwp_l<20 else("expansión" if bbwp_l>80 else "normal"))
        mcg_v=chart_data["mcg25"].iloc[-1]
        m4.metric("McGinley 25",f"{mcg_v:.2f}","↑ sobre MCG" if last_p>mcg_v else "↓ bajo MCG")
        e200_v=chart_data["ema200"].iloc[-1]
        m5.metric("EMA 200",f"{e200_v:.2f}","↑ sobre EMA" if last_p>e200_v else "↓ bajo EMA")

        with st.spinner(f"Renderizando {chart_ticker}…"):
            fig = build_figure(chart_data, chart_ticker, n_candles=zoom_candles)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("#### Señales actuales")
        sigs=build_signals(chart_data)
        pct_s,label_s,bull_n,total_n=score_signals(sigs)
        cols_sig=st.columns(len(sigs))
        for col,s in zip(cols_sig,sigs):
            emoji="🟢" if s["state"]=="bull" else("🔴" if s["state"]=="bear" else "⚪")
            col.markdown(f"<div style='text-align:center;font-size:.7rem;color:#fff'>{emoji}<br>{s['label']}</div>",
                         unsafe_allow_html=True)
        score_col="green" if pct_s>=60 else("red" if pct_s<40 else "orange")
        st.markdown(f"<h4 style='color:{score_col};text-align:center'>{label_s}  ·  {bull_n}/{total_n}  ({pct_s}%)</h4>",
                    unsafe_allow_html=True)

        buf=io.BytesIO()
        fig_dl=build_figure(chart_data,chart_ticker,n_candles=zoom_candles)
        fig_dl.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor=STYLE["bg"])
        plt.close(fig_dl); buf.seek(0)
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
    st.caption("Seguimiento de posiciones abiertas con semáforo de salida + confirmación Sniper.")

    with st.expander("➕ Añadir nueva posición", expanded=False):
        fc1,fc2,fc3 = st.columns(3)
        with fc1:
            new_ticker = st.selectbox("Ticker",options=sorted(ALL_TICKERS),key="new_ticker")
            new_divisa = st.text_input("Divisa (autodetectada)",value=detectar_divisa(new_ticker),key="new_divisa")
        with fc2:
            new_precio  = st.number_input("Precio de entrada",min_value=0.0,value=0.0,step=0.01,format="%.4f",key="new_precio")
            new_capital = st.number_input("Capital invertido",min_value=0.0,value=1000.0,step=100.0,key="new_capital")
        with fc3:
            new_fecha = st.date_input("Fecha de entrada",value=datetime.now(timezone.utc).date(),key="new_fecha")
            new_notas = st.text_input("Notas (opcional)",key="new_notas")
        if st.button("✅ Añadir posición",key="btn_añadir"):
            if new_ticker and new_precio>0:
                añadir_posicion(new_ticker,new_divisa,new_precio,str(new_fecha),new_capital,new_notas)
                st.success(f"✅ {new_ticker} añadido."); st.rerun()
            else:
                st.error("Introduce un ticker y precio válido.")

    df_cartera = cargar_cartera()
    if df_cartera.empty:
        st.info("No tienes posiciones abiertas. Añade una posición arriba. 👆")
    else:
        st.markdown(f"### Posiciones abiertas ({len(df_cartera)})")
        st.caption(f"🕐 {datetime.now(timezone.utc).strftime('%H:%M UTC')}")

        rows_display = []
        for idx, row in df_cartera.iterrows():
            ticker        = row["Ticker"]
            precio_entrada = float(row["Precio_Entrada"])
            capital       = float(row["Capital"])
            divisa        = row["Divisa"]
            fecha_entrada = row["Fecha_Entrada"]

            precio_actual = get_precio_actual(ticker)
            if precio_actual <= 0: precio_actual = precio_entrada
            pnl_pct    = (precio_actual-precio_entrada)/precio_entrada*100
            pnl_divisa = capital*pnl_pct/100
            try:
                dias=(datetime.now(timezone.utc).date()-pd.to_datetime(fecha_entrada).date()).days
            except Exception: dias="—"

            with st.spinner(f"Señales {ticker}…"):
                chart_d = cached_chart_data(ticker, interval_key="1D")

            if chart_d:
                sal = semaforo_salida(
                    df=chart_d["df"], kdf=chart_d["konc"], bitman_df=chart_d["bitman"],
                    macd_line=chart_d["macd_line"], macd_sig=chart_d["macd_sig"],
                    pvi_s=chart_d["pvi_s"], pvi_ema=chart_d["pvi_ema"])
                etiq_salida    = sal["etiqueta"]
                razones_salida = sal["razones"]
                sniper_cart_state = chart_d.get("sniper_state","IGNORAR")
                sniper_cart_velas = chart_d.get("sniper_velas",999)
            else:
                etiq_salida=razones_salida="⚪ Sin datos"
                sniper_cart_state="IGNORAR"; sniper_cart_velas=999

            rows_display.append({
                "idx":idx,"Ticker":ticker,"Divisa":divisa,
                "Entrada":f"{precio_entrada:.4f}","Actual":f"{precio_actual:.4f}",
                "P&L %":f"{pnl_pct:+.2f}%","P&L":f"{pnl_divisa:+.2f} {divisa}",
                "Días":dias,"Fecha entrada":fecha_entrada,
                "Semáforo":etiq_salida,"Sniper":sniper_cart_state,
                "Sniper_Velas":sniper_cart_velas,
                "Razones salida":razones_salida,"Notas":row.get("Notas",""),
            })

        df_display = pd.DataFrame(rows_display)
        cols_tabla = ["Ticker","Divisa","Entrada","Actual","P&L %","P&L","Días","Fecha entrada","Semáforo","Sniper"]
        _st=df_display[cols_tabla].style
        _fn=_st.map if hasattr(_st,"map") else _st.applymap
        s1=_fn(color_salida,subset=["Semáforo"])
        _fn2=s1.map if hasattr(s1,"map") else s1.applymap
        st.dataframe(
            _fn2(color_sniper,subset=["Sniper"])
            .set_properties(**{"background-color":"#13161e","color":"#ffffff","font-size":"0.82rem"})
            .set_table_styles([{"selector":"th","props":[
                ("background-color","#0d0f14"),("color","#efb030"),
                ("font-size","0.84rem"),("border-bottom","1px solid #1f2430")]}]),
            use_container_width=True, height=min(500,38+40*len(df_display)))

        st.markdown("---")
        st.markdown("### 🔍 Detalle por posición")
        for row_d in rows_display:
            ticker=row_d["Ticker"]; etiq=row_d["Semáforo"]
            snip_st=row_d["Sniper"]; snip_vl=row_d["Sniper_Velas"]
            col_etiq=SALIDA_COLOR.get(etiq,"#fff")
            with st.expander(
                f"{'🔴' if 'SALIDA' in etiq else '🟠' if 'REDUCIR' in etiq else '🟡' if 'VIGILAR' in etiq else '🟢'}"
                f"  {ticker}  —  P&L: {row_d['P&L %']}  ·  {etiq}"):
                dc1,dc2,dc3,dc4=st.columns(4)
                dc1.metric("Entrada",row_d["Entrada"]); dc2.metric("Actual",row_d["Actual"])
                dc3.metric("P&L %",row_d["P&L %"]);     dc4.metric("P&L",row_d["P&L"])
                st.markdown(f"**Semáforo salida:** <span style='color:{col_etiq};font-weight:bold'>{etiq}</span>",
                            unsafe_allow_html=True)
                st.caption(f"Señales: {row_d['Razones salida']}")
                st.markdown(f"**Confirmación Sniper:** {sniper_badge_html(snip_st,snip_vl)}",
                            unsafe_allow_html=True)
                if row_d["Notas"]: st.caption(f"📝 {row_d['Notas']}")
                st.markdown("**Cerrar posición:**")
                cc1,cc2=st.columns([2,1])
                with cc1:
                    precio_cierre=st.number_input("Precio de salida",min_value=0.0,
                        value=float(row_d["Actual"].replace(",",".")),step=0.01,format="%.4f",
                        key=f"cierre_{ticker}_{row_d['idx']}")
                with cc2:
                    st.markdown("<div style='height:28px'></div>",unsafe_allow_html=True)
                    if st.button(f"🔒 Cerrar {ticker}",key=f"btn_cerrar_{ticker}_{row_d['idx']}"):
                        cerrar_posicion(row_d["idx"],precio_cierre)
                        st.success(f"✅ {ticker} cerrado."); st.rerun()

        st.markdown("---")
        st.markdown("### 📊 Resumen")
        usd_rows=[r for r in rows_display if r["Divisa"]=="USD"]
        eur_rows=[r for r in rows_display if r["Divisa"]=="EUR"]
        def calcular_pnl_total(rows):
            return sum(float(r["P&L"].split()[0].replace(",",".")) for r in rows)
        rs1,rs2,rs3,rs4=st.columns(4)
        rs1.metric("Posiciones abiertas",len(rows_display))
        rs4.metric("🔥 Sniper activo",sum(1 for r in rows_display if r["Sniper"]=="ULTRA-SAFE SNIPER"))
        if usd_rows: rs2.metric("P&L total USD",f"{calcular_pnl_total(usd_rows):+.2f} $")
        if eur_rows: rs3.metric("P&L total EUR",f"{calcular_pnl_total(eur_rows):+.2f} €")

    st.markdown("---")
    st.markdown("### 📋 Historial de operaciones cerradas")
    df_hist=cargar_historial()
    if df_hist.empty:
        st.info("Aún no has cerrado ninguna posición.")
    else:
        def color_pnl(val):
            try:
                v=float(str(val).replace("%","").replace(",","."))
                return f"color:{'#26a65b' if v>=0 else '#e04040'};font-weight:bold"
            except Exception: return ""
        styler_h=df_hist.style; fn_h=styler_h.map if hasattr(styler_h,"map") else styler_h.applymap
        st.dataframe(fn_h(color_pnl,subset=["PnL_pct","PnL_divisa"])
                     .set_properties(**{"background-color":"#13161e","color":"#ffffff","font-size":"0.80rem"})
                     .set_table_styles([{"selector":"th","props":[
                         ("background-color","#0d0f14"),("color","#efb030"),("font-size","0.82rem")]}]),
                     use_container_width=True)
        if "PnL_pct" in df_hist.columns:
            ganadas=len(df_hist[df_hist["PnL_pct"]>0])
            perdidas=len(df_hist[df_hist["PnL_pct"]<=0])
            media=df_hist["PnL_pct"].mean()
            rh1,rh2,rh3,rh4=st.columns(4)
            rh1.metric("Operaciones cerradas",len(df_hist))
            rh2.metric("Ganadoras",ganadas)
            rh3.metric("Perdedoras",perdidas)
            rh4.metric("P&L medio",f"{media:+.2f}%")


