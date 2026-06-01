# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Sovereign Dashboard v3  —  Streamlit  —  2 pestañas
#   Tab 1 : Semáforo texto  (todos los tickers)
#   Tab 2 : Gráfico multi-panel  (ticker seleccionable)
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # backend sin GUI (obligatorio en servidor)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st

from ta.trend    import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator

# ── módulo propio con toda la lógica de indicadores ──────────────────────────
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
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title  = "Sovereign Dashboard v3",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ── CSS mínimo para tema oscuro + tablas ─────────────────────────────────────
st.markdown("""
<style>
    body, .stApp { background-color: #0d0f14; color: #c8cad0; }
    .stDataFrame thead th { background: #13161e !important; color: #efb030 !important; }
    .stDataFrame tbody tr:hover { background: #1f2430 !important; }
    div[data-testid="stTab"] button { font-weight: bold; font-size: 1rem; }
    .signal-badge {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 0.78rem; font-weight: bold; margin: 1px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PALETA (idéntica al bloque 2 original)
# ══════════════════════════════════════════════════════════════════════════════

STYLE = dict(
    bg="#0d0f14", panel="#13161e", border="#1f2430",
    bull="#26a65b", bear="#e04040",
    bull_fade="#26a65b55", bear_fade="#e0404055",
    mcg="#efb030", ema200="#6060dd",
    text="#c8cad0", muted="#555a6a",
    verde="#2ca85e", marron="#a06432",
    azul="#4488e0", media_k="#ffffff",
    pvi="#6090e0", pvi_ema="#efb030",
    macd_line="#6090e0", macd_sig="#efb030",
    rsi="#a78bfa", adx="#a78bfa",
    pdi="#26a65b", ndi="#e04040",
    ao_up="#26a65b", ao_dn="#e04040",
    grid="#1a1e28", zero="#2a2e3a",
)

plt.rcParams.update({
    "figure.facecolor": STYLE["bg"],
    "axes.facecolor":   STYLE["panel"],
    "axes.edgecolor":   STYLE["border"],
    "axes.labelcolor":  STYLE["muted"],
    "xtick.color":      STYLE["muted"],
    "ytick.color":      STYLE["muted"],
    "text.color":       STYLE["text"],
    "grid.color":       STYLE["grid"],
    "grid.linewidth":   0.5,
    "font.family":      "monospace",
    "font.size":        9,
})


# ══════════════════════════════════════════════════════════════════════════════
# LISTA DE TICKERS (idéntica al original)
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

# Grupos para filtrado rápido en Tab 1
GRUPOS = {
    "Todos":       ALL_TICKERS,
    "US Large Cap": ["AAPL","MSFT","AMZN","NVDA","GOOG","META","BRK-B","TSLA",
                     "JNJ","V","PG","XOM","UNH","JPM","HD","LLY","MA","CVX","ABBV",
                     "KO","PEP","COST","BAC","CRM","NFLX","ABT","MCD","LMT","EL",
                     "NEE","CAT","MRK"],
    "Tecnología":  ["AAPL","MSFT","NVDA","GOOG","META","TSLA","ADBE","AVGO","CSCO",
                    "AMD","TXN","QCOM","AMAT","LRCX","INTU","VRTX","ZS","PLTR","MU",
                    "LITE","ON","ASML","SAP","SIE.DE","IFX.DE","AI.PA"],
    "Europa":      ["MC.PA","SIE.DE","ENGI.PA","AIR.PA","ALV.DE","EL.PA","AI.PA",
                    "BNP.PA","SAN.PA","KER.PA","SU.PA","NESN.SW","LIN.DE","VOW3.DE",
                    "BMW.DE","ADS.DE","IFX.DE","MUV2.DE","FRE.DE","DTE.DE","RWE.DE",
                    "OR.PA","TTE"],
    "España":      ["ITX.MC","BBVA.MC","SAN.MC","TEF.MC","IBE.MC","REP.MC","FER.MC",
                    "ACX.MC","ACS.MC","AENA.MC","ANA.MC","IAG.MC","LOG.MC","MAP.MC",
                    "PUIG.MC","NTGY.MC","ELE.MC","IDR.MC"],
    "China / Asia":["PDD","NIO","TCEHY","BZUN","FUTU","MOMO","MNSO","TAL","EDU",
                    "WB","XPEV"],
    "Crypto / Materias": ["GC=F","SI=F","BTC-USD","ETH-USD","XRP-USD"],
}


# ══════════════════════════════════════════════════════════════════════════════
# CACHÉ — Tab 1: semáforo completo  (TTL 1 hora)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def cached_dashboard(tickers_tuple: tuple) -> pd.DataFrame:
    return get_sovereign_dashboard(list(tickers_tuple))


# ══════════════════════════════════════════════════════════════════════════════
# CACHÉ — Tab 2: datos + indicadores de un ticker  (TTL 1 hora)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def cached_chart_data(ticker: str, period: str = "2y") -> dict:
    """Descarga y calcula todos los indicadores del gráfico para un ticker."""
    df = download_df(ticker, period=period, interval="1d")
    if df.empty or len(df) < 60:
        return {}

    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    mcg25    = mcginley_dynamic(close, 25)
    ema200   = EMAIndicator(close=close, window=200).ema_indicator()
    adx_ind  = ADXIndicator(high=high, low=low, close=close, window=14)
    adx_s    = adx_ind.adx()
    pdi_s    = adx_ind.adx_pos()
    ndi_s    = adx_ind.adx_neg()
    ao_s     = awesome_osc(high, low)
    bitman   = clasificar_bitman(df)
    div_df   = detectar_divergencia_simple(df)
    bbw_s, bbwp_s = calculate_bbwp(close, bb_len=13, lookback=252)
    konc     = compute_blai5_koncorde(df, m=15)
    pvi_s    = calculate_pvi(close, volume)
    pvi_ema  = pvi_s.ewm(span=25, adjust=False).mean()
    macd_obj = MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    rsi_s    = RSIIndicator(close=close, window=14).rsi()

    return dict(
        df=df, mcg25=mcg25, ema200=ema200,
        adx_s=adx_s, pdi_s=pdi_s, ndi_s=ndi_s,
        ao_s=ao_s, bitman=bitman, div_df=div_df,
        bbwp_s=bbwp_s, konc=konc,
        pvi_s=pvi_s, pvi_ema=pvi_ema,
        macd_line=macd_obj.macd(),
        macd_sig=macd_obj.macd_signal(),
        macd_hist=macd_obj.macd_diff(),
        rsi_s=rsi_s,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE GRÁFICO
# ══════════════════════════════════════════════════════════════════════════════

def sv(series: pd.Series, index) -> np.ndarray:
    """Alinea una serie al índice recortado."""
    return series.reindex(index).values


def format_xaxis(ax, index, n_labels: int = 8):
    step   = max(1, len(index) // n_labels)
    ticks  = list(range(0, len(index), step))
    labels = [index[i].strftime("%d %b") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7)


def panel_style(ax, ylabel: str = "", yticks: int = 5, zero_line: bool = False):
    ax.set_ylabel(ylabel, fontsize=8, labelpad=4)
    ax.yaxis.set_major_locator(plt.MaxNLocator(yticks, prune="both"))
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, axis="y", linewidth=0.4)
    ax.grid(True, axis="x", linewidth=0.2, alpha=0.4)
    if zero_line:
        ax.axhline(0, color=STYLE["zero"], linewidth=0.8, zorder=1)
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["border"])


# ══════════════════════════════════════════════════════════════════════════════
# SEÑALES RESUMEN  (para barra inferior del gráfico)
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

    p = close.iloc[-1]

    def sig(label, bull, neutral=False):
        return {"label": label, "state": "neutral" if neutral else ("bull" if bull else "bear")}

    sigs = [
        sig(f"precio {'>' if p >= mcg25.iloc[-1] else '<'} MCG25",   p >= mcg25.iloc[-1]),
        sig(f"precio {'>' if p >= ema200.iloc[-1] else '<'} EMA200", p >= ema200.iloc[-1]),
    ]
    r = rsi_s.iloc[-1]
    sigs.append(sig(f"RSI {r:.1f}", r > 50, neutral=45 < r < 55))
    sigs.append(sig(f"MACD hist {'↑' if macd_h.iloc[-1] >= 0 else '↓'}", macd_h.iloc[-1] >= 0))
    sigs.append(sig(f"MACD línea {'≥0' if macd_l.iloc[-1] >= 0 else '<0'}", macd_l.iloc[-1] >= 0))

    if not konc.empty:
        sigs.append(sig(f"Azul Konc {'↑' if konc['azul'].iloc[-1] >= 0 else '↓'}",
                        konc["azul"].iloc[-1] >= 0))
        sigs.append(sig(f"Verde {'>' if konc['verde'].iloc[-1] >= konc['marron'].iloc[-1] else '<'} Marrón",
                        konc["verde"].iloc[-1] >= konc["marron"].iloc[-1]))

    sigs.append(sig(f"PVI {'>' if pvi_s.iloc[-1] >= pvi_ema.iloc[-1] else '<'} EMA25",
                    pvi_s.iloc[-1] >= pvi_ema.iloc[-1]))

    a = adx_s.iloc[-1]
    sigs.append(sig(f"ADX {a:.1f} {'fuerte' if a > 25 else 'débil'}", a > 25, neutral=18 < a < 25))

    if bitman is not None and not bitman.empty:
        b_etiq  = bitman["Bitman_Etiqueta"].iloc[-1]
        b_v     = int(bitman["Bitman_Velas"].iloc[-1])
        b_bull  = b_etiq in ("IMPULSO ALCISTA", "RETROCESO ALCISTA")
        sigs.append(sig(f"Bitman {b_etiq[:8]} ({b_v}v)", b_bull, neutral=(b_etiq == "INDEFINICIÓN")))

    if div_df is not None:
        dt = div_df["divergencia_tipo"].iloc[-1]
        if dt == "alcista":   sigs.append({"label": "Div RSI alcista", "state": "bull"})
        elif dt == "bajista": sigs.append({"label": "Div RSI bajista", "state": "bear"})

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
# GRÁFICO MULTI-PANEL  (fiel al bloque 2 original)
# ══════════════════════════════════════════════════════════════════════════════

def build_figure(data: dict, ticker: str) -> plt.Figure:
    df       = data["df"]
    close    = df["Close"]
    high     = df["High"]
    low      = df["Low"]
    mcg25    = data["mcg25"]
    ema200   = data["ema200"]
    adx_s    = data["adx_s"]
    pdi_s    = data["pdi_s"]
    ndi_s    = data["ndi_s"]
    ao_s     = data["ao_s"]
    bitman   = data["bitman"]
    div_df   = data["div_df"]
    bbwp_s   = data["bbwp_s"]
    konc     = data["konc"]
    pvi_s    = data["pvi_s"]
    pvi_ema  = data["pvi_ema"]
    macd_line= data["macd_line"]
    macd_sig = data["macd_sig"]
    macd_hist= data["macd_hist"]
    rsi_s    = data["rsi_s"]

    sigs               = build_signals(data)
    pct, score_label, bull_n, total_n = score_signals(sigs)
    score_color = STYLE["bull"] if pct >= 60 else (STYLE["bear"] if pct < 40 else STYLE["mcg"])

    # ── recorte a 252 velas ────────────────────────────────────────────────
    n_max   = min(252, len(df))
    df_plot = df.iloc[-n_max:]
    idx     = df_plot.index
    xs      = np.arange(len(idx))

    # ── figura ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 20), facecolor=STYLE["bg"])
    fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.03, hspace=0.06)
    heights = [5, 2, 2.2, 1.4, 1.6, 1.8, 1.6]
    gs      = gridspec.GridSpec(7, 1, figure=fig, height_ratios=heights, hspace=0.06)
    axes    = [fig.add_subplot(gs[i]) for i in range(7)]
    for i in range(6):
        axes[i].tick_params(labelbottom=False)

    # título
    last_p  = close.iloc[-1]
    prev_p  = close.iloc[-2]
    chg     = last_p - prev_p
    pct_chg = chg / prev_p * 100
    chg_c   = STYLE["bull"] if chg >= 0 else STYLE["bear"]
    sign    = "+" if chg >= 0 else ""
    fig.text(0.07, 0.965, ticker,
             fontsize=18, fontweight="bold", color=STYLE["text"], va="bottom")
    fig.text(0.19, 0.965, f"{last_p:.2f}",
             fontsize=16, fontweight="bold", color=STYLE["text"], va="bottom")
    fig.text(0.30, 0.965, f"{sign}{chg:.2f}  ({sign}{pct_chg:.2f}%)",
             fontsize=12, color=chg_c, va="bottom")
    fig.text(0.97, 0.965, f"{score_label}  ·  {bull_n}/{total_n}  ({pct}%)",
             fontsize=11, color=score_color, ha="right", va="bottom", style="italic")

    # ── PANEL 0 — Velas + MCG25 + EMA200 ──────────────────────────────────
    ax0 = axes[0]
    w   = 0.4
    for i, (_, row) in enumerate(df_plot.iterrows()):
        col = STYLE["bull"] if row["Close"] >= row["Open"] else STYLE["bear"]
        ax0.plot([i, i], [row["Low"], row["High"]], color=col, lw=0.8, zorder=2)
        ax0.add_patch(plt.Rectangle(
            (i - w, min(row["Open"], row["Close"])),
            2 * w, max(abs(row["Close"] - row["Open"]), 0.001),
            color=col, zorder=3))
    ax0.plot(xs, sv(mcg25,  idx), color=STYLE["mcg"],   lw=1.4, label="MCG 25",  zorder=4)
    ax0.plot(xs, sv(ema200, idx), color=STYLE["ema200"], lw=1.4, label="EMA 200", zorder=4)
    ax0.set_xlim(-1, len(idx))
    ax0.legend(loc="upper left", fontsize=8, frameon=False,
               labelcolor=[STYLE["mcg"], STYLE["ema200"]])
    panel_style(ax0, ylabel="Precio")
    ax0.set_title("Velas  ·  McGinley 25  ·  EMA 200",
                  fontsize=9, color=STYLE["muted"], loc="right", pad=4)
    format_xaxis(ax0, idx)
    ax0.tick_params(labelbottom=True)

    # ── PANEL 1 — ADX + AO ────────────────────────────────────────────────
    ax1  = axes[1]
    ax1r = ax1.twinx()
    ao_v = sv(ao_s, idx)
    ao_p = np.roll(ao_v, 1); ao_p[0] = ao_v[0]
    ao_c = [STYLE["ao_up"] if ao_v[i] >= ao_p[i] else STYLE["ao_dn"] for i in range(len(ao_v))]
    ax1r.bar(xs, ao_v, color=ao_c, alpha=0.7, width=0.8, zorder=2)
    ax1r.axhline(0, color=STYLE["zero"], lw=0.7)
    ax1r.tick_params(labelsize=7, colors=STYLE["muted"])
    ax1r.set_ylabel("AO", fontsize=7, color=STYLE["muted"])
    ax1r.spines["right"].set_edgecolor(STYLE["border"])
    ax1.plot(xs, sv(adx_s, idx), color=STYLE["adx"], lw=1.4, label="ADX", zorder=3)
    ax1.plot(xs, sv(pdi_s, idx), color=STYLE["pdi"], lw=0.9, ls="--", label="+DI", zorder=3)
    ax1.plot(xs, sv(ndi_s, idx), color=STYLE["ndi"], lw=0.9, ls="--", label="-DI", zorder=3)
    ax1.axhline(25, color=STYLE["muted"], lw=0.6, ls=":")
    ax1.legend(loc="upper left", fontsize=7, frameon=False,
               labelcolor=[STYLE["adx"], STYLE["pdi"], STYLE["ndi"]])
    panel_style(ax1, ylabel="ADX")
    ax1.set_title("ADX  ·  +DI / -DI  ·  Awesome Oscillator",
                  fontsize=9, color=STYLE["muted"], loc="right", pad=4)

    # ── PANEL 2 — Koncorde ────────────────────────────────────────────────
    ax2 = axes[2]
    if not konc.empty:
        for key, col in [("verde", STYLE["verde"]), ("marron", STYLE["marron"]),
                         ("azul", STYLE["azul"])]:
            v = sv(konc[key], idx)
            ax2.fill_between(xs, v, alpha=0.40, color=col, label=key.capitalize(), zorder=2)
            ax2.plot(xs, v, color=col, lw=1.0, zorder=3)
        ax2.plot(xs, sv(konc["media"], idx), color=STYLE["media_k"],
                 lw=1.6, label="Media", zorder=4)
        ax2.legend(loc="upper left", fontsize=7, frameon=False,
                   labelcolor=[STYLE["verde"], STYLE["marron"],
                               STYLE["azul"], STYLE["media_k"]])
    ax2.axhline(0, color=STYLE["zero"], lw=0.7)
    panel_style(ax2, ylabel="Koncorde")
    ax2.set_title("Blai5 Koncorde  ·  Verde / Marrón / Azul / Media",
                  fontsize=9, color=STYLE["muted"], loc="right", pad=4)

    # ── PANEL 3 — BBWP 13/252 ─────────────────────────────────────────────
    ax3    = axes[3]
    bbwp_v = sv(bbwp_s, idx)
    ax3.fill_between(xs, bbwp_v, 20,
                     where=(~np.isnan(bbwp_v)) & (bbwp_v < 20),
                     alpha=0.20, color=STYLE["azul"], zorder=1)
    ax3.fill_between(xs, bbwp_v, 80,
                     where=(~np.isnan(bbwp_v)) & (bbwp_v > 80),
                     alpha=0.20, color=STYLE["bear"], zorder=1)
    for i in range(1, len(xs)):
        if np.isnan(bbwp_v[i]) or np.isnan(bbwp_v[i-1]): continue
        mid = (bbwp_v[i] + bbwp_v[i-1]) / 2
        lc  = STYLE["azul"] if mid < 20 else (STYLE["bear"] if mid > 80 else STYLE["muted"])
        ax3.plot([xs[i-1], xs[i]], [bbwp_v[i-1], bbwp_v[i]], color=lc, lw=1.5, zorder=3)
    ax3.axhline(80, color=STYLE["bear"],  lw=0.7, ls="--", alpha=0.6)
    ax3.axhline(20, color=STYLE["azul"],  lw=0.7, ls="--", alpha=0.6)
    ax3.axhline(50, color=STYLE["muted"], lw=0.5, ls=":",  alpha=0.4)
    ax3.set_ylim(-2, 102)
    ax3.yaxis.set_ticks([0, 20, 50, 80, 100])
    panel_style(ax3, ylabel="BBWP")
    ax3.set_title("BBWP 13/252  ·  🟢 compresión < 20  ·  🔴 expansión > 80",
                  fontsize=9, color=STYLE["muted"], loc="right", pad=4)

    # ── PANEL 4 — PVI + EMA25 ─────────────────────────────────────────────
    ax4   = axes[4]
    pvi_v = sv(pvi_s,   idx)
    pvi_e = sv(pvi_ema, idx)
    ax4.fill_between(xs, pvi_v, pvi_e, where=(pvi_v >= pvi_e), alpha=0.18, color=STYLE["bull"])
    ax4.fill_between(xs, pvi_v, pvi_e, where=(pvi_v <  pvi_e), alpha=0.18, color=STYLE["bear"])
    ax4.plot(xs, pvi_v, color=STYLE["pvi"],     lw=1.4, label="PVI")
    ax4.plot(xs, pvi_e, color=STYLE["pvi_ema"], lw=1.4, ls="--", label="EMA 25")
    ax4.legend(loc="upper left", fontsize=7, frameon=False,
               labelcolor=[STYLE["pvi"], STYLE["pvi_ema"]])
    panel_style(ax4, ylabel="PVI")
    ax4.set_title("PVI  ·  EMA 25", fontsize=9, color=STYLE["muted"], loc="right", pad=4)

    # ── PANEL 5 — MACD ────────────────────────────────────────────────────
    ax5     = axes[5]
    hist_v  = sv(macd_hist, idx)
    hist_p  = np.roll(hist_v, 1); hist_p[0] = hist_v[0]
    bar_col = []
    for i in range(len(hist_v)):
        v, p_v = hist_v[i], hist_p[i]
        if np.isnan(v): bar_col.append(STYLE["muted"]); continue
        if v >= 0: bar_col.append(STYLE["bull"] if v >= p_v else STYLE["bull_fade"])
        else:      bar_col.append(STYLE["bear"] if v <= p_v else STYLE["bear_fade"])
    ax5.bar(xs, hist_v, color=bar_col, width=0.8, alpha=0.9, zorder=2)
    ax5.plot(xs, sv(macd_line, idx), color=STYLE["macd_line"], lw=1.3, label="MACD",  zorder=3)
    ax5.plot(xs, sv(macd_sig,  idx), color=STYLE["macd_sig"],  lw=1.3, ls="--", label="Señal", zorder=3)
    ax5.axhline(0, color=STYLE["zero"], lw=0.7)
    ax5.legend(loc="upper left", fontsize=7, frameon=False,
               labelcolor=[STYLE["macd_line"], STYLE["macd_sig"]])
    panel_style(ax5, ylabel="MACD")
    ax5.set_title("MACD  12 / 26 / 9", fontsize=9, color=STYLE["muted"], loc="right", pad=4)

    # ── PANEL 6 — RSI + divergencias ──────────────────────────────────────
    ax6   = axes[6]
    rsi_v = sv(rsi_s, idx)
    ax6.fill_between(xs, rsi_v, 70, where=(rsi_v > 70), alpha=0.25, color=STYLE["bull"])
    ax6.fill_between(xs, rsi_v, 30, where=(rsi_v < 30), alpha=0.25, color=STYLE["bear"])
    ax6.plot(xs, rsi_v, color=STYLE["rsi"], lw=1.4)
    for lvl, col, ls in [(70, STYLE["bear"], "--"), (50, STYLE["muted"], ":"),
                         (30, STYLE["bull"], "--")]:
        ax6.axhline(lvl, color=col, lw=0.7, ls=ls)
    if div_df is not None:
        rsi_aligned = sv(rsi_s, idx)
        div_tipos   = sv(div_df["divergencia_tipo"].map(
            lambda x: x if isinstance(x, str) else "ninguna"), idx)
        for xi_d, (dt, rv) in enumerate(zip(div_tipos, rsi_aligned)):
            if dt == "alcista":
                ax6.annotate("▲", xy=(xi_d, rv), fontsize=8, color=STYLE["bull"],
                             ha="center", va="top",
                             xytext=(0, -8), textcoords="offset points")
            elif dt == "bajista":
                ax6.annotate("▼", xy=(xi_d, rv), fontsize=8, color=STYLE["bear"],
                             ha="center", va="bottom",
                             xytext=(0, 8), textcoords="offset points")
    ax6.set_ylim(0, 100)
    ax6.yaxis.set_ticks([30, 50, 70])
    panel_style(ax6, ylabel="RSI", yticks=3)
    format_xaxis(ax6, idx)
    ax6.tick_params(labelbottom=True)
    ax6.set_title("RSI  14  ·  ▲ div alcista  ▼ div bajista",
                  fontsize=9, color=STYLE["muted"], loc="right", pad=4)

    for ax in axes:
        ax.set_xlim(-1, len(idx))

    # ── recuadro informativo ──────────────────────────────────────────────
    _precio   = close.iloc[-1]
    _mcg_val  = mcg25.iloc[-1]
    _e200_val = ema200.iloc[-1]
    _rsi_val  = rsi_s.iloc[-1]

    _ak = "🟢" if (not konc.empty and konc["azul"].iloc[-1] > 0)  else "🔴"
    if not konc.empty:
        _area_max  = konc[["verde", "marron", "azul"]].max(axis=1)
        _area_min  = konc[["verde", "marron", "azul"]].min(axis=1)
        _media_val = konc["media"].iloc[-1]
        _pk = "🟢" if (not pd.isna(_media_val) and
                       _area_min.iloc[-1] <= _media_val <= _area_max.iloc[-1]) else "🔴"
    else:
        _pk = "⚪"

    _pvi_str = "🟢 PVI>" if pvi_s.iloc[-1] > pvi_ema.iloc[-1] else "🔴 PVI<"

    if bitman is not None and not bitman.empty:
        _b_etiq  = bitman["Bitman_Etiqueta"].iloc[-1]
        _b_velas = int(bitman["Bitman_Velas"].iloc[-1])
        _b_e     = ("📈" if _b_etiq in ("IMPULSO ALCISTA", "RETROCESO ALCISTA")
                    else ("📉" if "BAJISTA" in _b_etiq else "⬜"))
        _bitman_str = f"{_b_etiq} ({_b_velas}v) {_b_e}"
    else:
        _bitman_str = "N/D"

    if div_df is not None:
        _hits = div_df[div_df["divergencia_tipo"] != "ninguna"]
        if not _hits.empty:
            _dlast  = _hits.iloc[-1]["divergencia_tipo"]
            _didx   = div_df.index.get_loc(_hits.index[-1])
            _dv     = len(div_df) - 1 - _didx
            _de     = "🟢" if _dlast == "alcista" else "🔴"
            if _dv <= 5:    _div_str = f"{_de} {_dlast.upper()} FRESCA ({_dv}v)"
            elif _dv <= 20: _div_str = f"{_de} {_dlast.upper()} válida ({_dv}v)"
            elif _dv <= 50: _div_str = f"{'🟡' if _dlast=='alcista' else '🟠'} ctx ({_dv}v)"
            else:           _div_str = f"⚪ caducada ({_dv}v)"
        else:
            _div_str = "⚪ sin divergencia"
    else:
        _div_str = "⚪"

    _bbwp_v   = bbwp_s.dropna().iloc[-1] if len(bbwp_s.dropna()) > 0 else np.nan
    _bbwp_str = (f"{'🟢' if _bbwp_v < 20 else ('🔴' if _bbwp_v > 80 else '⚪')} {_bbwp_v:.1f}%  (13/252)"
                 if not np.isnan(_bbwp_v) else "⚪ n/d")

    _mcg_sym  = "🟡" if abs(_precio / _mcg_val  - 1) < 0.012 else ("🟢" if _precio > _mcg_val  else "🔴")
    _e200_sym = "🟡" if abs(_precio / _e200_val - 1) < 0.015 else ("🟢" if _precio > _e200_val else "🔴")

    _lines = [
        f"Tendencia  MCG25:{_mcg_sym}  EMA200:{_e200_sym}   RSI:{_rsi_val:.1f}",
        f"Koncorde   Azul:{_ak}  Punto:{_pk}",
        f"PVI        {_pvi_str} EMA25",
        f"BBWP       {_bbwp_str}",
        f"Bitman     {_bitman_str}",
        f"Div RSI    {_div_str}",
        f"SCORE      {score_label}  ·  {bull_n}/{total_n}  ({pct}%)",
    ]

    _bx = fig.add_axes([0.07, 0.870 - 0.095, 0.27, 0.095], frameon=True)
    _bx.set_facecolor(STYLE["panel"])
    for sp in _bx.spines.values():
        sp.set_edgecolor(STYLE["border"]); sp.set_linewidth(0.8)
    _bx.set_xticks([]); _bx.set_yticks([])
    for li, line in enumerate(_lines):
        _bx.text(0.03, 0.94 - li * 0.135, line,
                 transform=_bx.transAxes,
                 fontsize=7,
                 color=score_color if li == 6 else STYLE["text"],
                 va="top", family="monospace")

    # ── barra de señales ──────────────────────────────────────────────────
    x0  = 0.07
    gap = (0.97 - x0) / len(sigs)
    for i, s in enumerate(sigs):
        col = (STYLE["bull"] if s["state"] == "bull" else
               STYLE["bear"] if s["state"] == "bear" else STYLE["muted"])
        fig.text(x0 + i * gap + (gap - 0.003) / 2, 0.025,
                 s["label"], fontsize=7.5, color=col,
                 ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor=STYLE["panel"],
                           edgecolor=col + "66", linewidth=0.8))

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ESTILO DE TABLA — Tab 1
# ══════════════════════════════════════════════════════════════════════════════

SEÑAL_COLOR = {
    "🚀 COMPRA 100%":       "#26a65b",
    "🟡 COMPRA 50%":        "#efb030",
    "⚠️ ATENCIÓN KONKORDE": "#efb030",
    "👀 VIGILAR":           "#6090e0",
    "⏰ LLEGAS TARDE":      "#888888",
    "⚠️ VIGILAR SALIDA":   "#e08030",
    "🔴 VENTA":             "#e04040",
    "⛔ SIN SETUP":         "#555555",
    "⛔ NI DE COÑA":        "#e04040",
}


def color_señal(val: str) -> str:
    col = SEÑAL_COLOR.get(val, "#c8cad0")
    return f"color: {col}; font-weight: bold"


def style_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    return (
        df.style
          .applymap(color_señal, subset=["Señal"])
          .set_properties(**{
              "background-color": "#13161e",
              "color": "#c8cad0",
              "border-color": "#1f2430",
              "font-size": "0.82rem",
          })
          .set_table_styles([{
              "selector": "th",
              "props": [("background-color", "#0d0f14"),
                        ("color", "#efb030"),
                        ("font-size", "0.85rem"),
                        ("border-bottom", "1px solid #1f2430")],
          }])
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  —  controles globales
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.markdown("---")

    # -- Tab 1 controls
    st.markdown("### 📋 Tab 1 — Semáforo")
    grupo_sel = st.selectbox(
        "Grupo de tickers",
        options=list(GRUPOS.keys()),
        index=0,
        key="grupo",
    )
    custom_input = st.text_area(
        "Tickers personalizados (uno por línea, vacío = usar grupo)",
        height=120,
        key="custom_tickers",
    )
    force_refresh_tab1 = st.button("🔄 Recalcular semáforo", key="refresh1")

    st.markdown("---")
    st.markdown("### 📈 Tab 2 — Gráfico")
    chart_ticker = st.selectbox(
        "Ticker para gráfico",
        options=sorted(ALL_TICKERS),
        index=ALL_TICKERS.index("NVDA") if "NVDA" in ALL_TICKERS else 0,
        key="chart_ticker",
    )
    chart_period = st.selectbox(
        "Período",
        options=["1y", "2y", "5y"],
        index=1,
        key="chart_period",
    )
    force_refresh_tab2 = st.button("🔄 Recalcular gráfico", key="refresh2")

    st.markdown("---")
    st.markdown(
        "<small>Datos: Yahoo Finance  ·  Caché 1h<br>"
        "BBWP config 13/252 estructural</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — dos pestañas
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs([
    "📋  Semáforo  —  Sovereign Dashboard v3",
    "📈  Gráfico multi-panel",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 : Semáforo texto
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("## 📋 Sovereign Dashboard v3 — Semáforo + Velas")
    st.caption(
        "Señales ordenadas por prioridad. "
        "🚀 COMPRA 100% = 5/5 🟢 con MACD y Media ≤5v + Bitman impulso.  "
        "Velas⏱: M=MACD · Az=Azul K · Me=Media K · B=Bitman fresco"
    )

    # ── selección de tickers ──────────────────────────────────────────────
    if custom_input.strip():
        tickers_tab1 = [t.strip().upper() for t in custom_input.strip().splitlines() if t.strip()]
    else:
        tickers_tab1 = GRUPOS[grupo_sel]

    col_meta1, col_meta2, col_meta3 = st.columns(3)
    col_meta1.metric("Tickers seleccionados", len(tickers_tab1))

    # ── carga con progreso ────────────────────────────────────────────────
    cache_key = tuple(sorted(tickers_tab1))

    if force_refresh_tab1:
        # forzar recálculo eliminando la entrada del caché
        cached_dashboard.clear()

    with st.spinner(f"Calculando {len(tickers_tab1)} tickers…"):
        progress_bar = st.progress(0)
        status_text  = st.empty()

        # Usamos un contenedor mutable para el callback
        _progress_state = {"i": 0}

        def _cb(i, total, ticker_name):
            _progress_state["i"] = i
            pct_val = int(i / total * 100)
            progress_bar.progress(pct_val)
            status_text.caption(f"⏳ Procesando {ticker_name}  ({i}/{total})")

        # Si ya está en caché, el callback no se llama (instantáneo)
        df_result = cached_dashboard(cache_key)

        progress_bar.progress(100)
        status_text.empty()

    if df_result.empty:
        st.warning("No se obtuvieron resultados. Revisa la conexión o los tickers.")
    else:
        col_meta2.metric("Tickers procesados", len(df_result))
        bulls_count = len(df_result[df_result["Señal"].isin(
            ["🚀 COMPRA 100%", "🟡 COMPRA 50%", "⚠️ ATENCIÓN KONKORDE"]
        )])
        col_meta3.metric("Señales alcistas", bulls_count)

        # ── filtros de señal ─────────────────────────────────────────────
        señales_unicas = ["Todas"] + list(df_result["Señal"].unique())
        filtro_señal   = st.multiselect(
            "Filtrar por señal",
            options=señales_unicas[1:],
            default=[],
            key="filtro_señal",
        )
        df_show = df_result[df_result["Señal"].isin(filtro_señal)] if filtro_señal else df_result

        # ── tabla principal ──────────────────────────────────────────────
        # Columna Razones es muy larga: la mostramos en expandible
        cols_tabla  = [c for c in df_show.columns if c != "Razones"]
        cols_razon  = ["Ticker", "Señal", "Razones"]

        st.dataframe(
            style_df(df_show[cols_tabla]),
            use_container_width=True,
            height=min(600, 38 + 35 * len(df_show)),
        )

        with st.expander("📝 Ver Razones detalladas"):
            st.dataframe(
                df_show[cols_razon].style.applymap(color_señal, subset=["Señal"]),
                use_container_width=True,
            )

        # ── leyenda ──────────────────────────────────────────────────────
        st.markdown("""
---
**Leyenda de señales**

| Señal | Condición |
|---|---|
| 🚀 COMPRA 100% | 5/5 🟢 + MACD y Media ≤5v + Bitman impulso alcista |
| 🟡 COMPRA 50%  | 4/5 o 5/5 🟢 con confluencia fresca |
| ⚠️ ATENCIÓN KONKORDE | Verde K < 0 y Azul K > 0 — señal independiente |
| ⏰ LLEGAS TARDE | Condiciones activas pero señal > 5v |
| ⚠️ VIGILAR SALIDA | 3+/5 activos pero c1 o c3 empezando a girar |
| 🔴 VENTA | Mayoría de condiciones desactivadas |
| 👀 VIGILAR | 3/5 activos — esperar confirmación |
| ⛔ NI DE COÑA | ≤1 activo y ≥3 rojos |
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 : Gráfico multi-panel
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown(f"## 📈 Gráfico multi-panel — **{chart_ticker}**")
    st.caption(
        "7 paneles: Velas+MCG25+EMA200 · ADX+AO · Koncorde · BBWP 13/252 · "
        "PVI+EMA25 · MACD 12/26/9 · RSI14+divergencias"
    )

    if force_refresh_tab2:
        cached_chart_data.clear()

    with st.spinner(f"Calculando indicadores para {chart_ticker}…"):
        chart_data = cached_chart_data(chart_ticker, period=chart_period)

    if not chart_data:
        st.error(f"No se pudieron obtener datos para **{chart_ticker}**. "
                 "Verifica el ticker y la conexión.")
    else:
        # ── métricas rápidas ─────────────────────────────────────────────
        df_c   = chart_data["df"]
        close  = df_c["Close"]
        m1, m2, m3, m4, m5 = st.columns(5)

        last_p = close.iloc[-1]
        prev_p = close.iloc[-2]
        chg    = last_p - prev_p
        pct_c  = chg / prev_p * 100
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
        m4.metric("McGinley 25",  f"{mcg_v:.2f}",
                  "↑ precio sobre" if last_p > mcg_v else "↓ precio bajo")

        e200_v = chart_data["ema200"].iloc[-1]
        m5.metric("EMA 200", f"{e200_v:.2f}",
                  "↑ precio sobre" if last_p > e200_v else "↓ precio bajo")

        # ── figura matplotlib ────────────────────────────────────────────
        with st.spinner("Renderizando gráfico…"):
            fig = build_figure(chart_data, chart_ticker)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)   # liberar memoria

        # ── barra de señales en Streamlit (adicional al gráfico) ─────────
        st.markdown("#### Señales actuales")
        sigs = build_signals(chart_data)
        pct_s, label_s, bull_n, total_n = score_signals(sigs)

        cols_sig = st.columns(len(sigs))
        for col, s in zip(cols_sig, sigs):
            color = ("🟢" if s["state"] == "bull" else
                     "🔴" if s["state"] == "bear" else "⚪")
            col.markdown(
                f"<div style='text-align:center;font-size:0.7rem;'>"
                f"{color}<br>{s['label']}</div>",
                unsafe_allow_html=True,
            )

        score_col = "green" if pct_s >= 60 else ("red" if pct_s < 40 else "orange")
        st.markdown(
            f"<h4 style='color:{score_col};text-align:center;'>"
            f"{label_s}  ·  {bull_n}/{total_n}  ({pct_s}%)</h4>",
            unsafe_allow_html=True,
        )

        # ── botón de descarga ────────────────────────────────────────────
        import io
        buf = io.BytesIO()
        fig_dl = build_figure(chart_data, chart_ticker)
        fig_dl.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                       facecolor=STYLE["bg"])
        plt.close(fig_dl)
        buf.seek(0)
        st.download_button(
            label="⬇️ Descargar gráfico PNG",
            data=buf,
            file_name=f"sovereign_{chart_ticker.replace('=','').replace('-','_')}.png",
            mime="image/png",
        )
