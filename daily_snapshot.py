# scripts/daily_snapshot.py
# ─────────────────────────────────────────────────────────────────────────────
# Script de CLI para GitHub Actions:
#   - Genera snapshots/YYYY-MM-DD_semaforo.csv
#   - Genera snapshots/YYYY-MM-DD_NVDA.png   (y cualquier CHART_TICKERS)
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import sys, os
from pathlib import Path
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# asegura que el raíz del repo esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators import (
    get_sovereign_dashboard,
    download_df,
    mcginley_dynamic, calculate_pvi,
    compute_blai5_koncorde, blai5_signals,
    clasificar_bitman, detectar_divergencia_simple,
    calculate_bbwp,
)
from ta.trend    import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator

# ── importamos las funciones de gráfico desde app.py ─────────────────────────
# (evitamos importar streamlit; build_figure no depende de él)
import importlib.util, types

def _load_app_functions():
    """Carga build_figure y build_signals desde app.py sin iniciar Streamlit."""
    spec   = importlib.util.spec_from_file_location(
        "app_module",
        Path(__file__).resolve().parent.parent / "app.py",
    )
    module = types.ModuleType("app_module")
    # parchear streamlit antes de ejecutar app.py
    import unittest.mock as mock
    with mock.patch.dict("sys.modules", {"streamlit": mock.MagicMock()}):
        spec.loader.exec_module(module)
    return module.build_figure, module.build_signals, module.score_signals, module.cached_chart_data.__wrapped__


# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────

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

# Tickers para los que se genera PNG diario
CHART_TICKERS = ["NVDA", "AAPL", "BTC-USD", "BBVA.MC"]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    today     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snap_dir  = Path("snapshots") / today
    snap_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Semáforo CSV ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Sovereign Dashboard — snapshot {today}")
    print(f"  Tickers: {len(ALL_TICKERS)}")
    print(f"{'='*60}\n")

    df_result = get_sovereign_dashboard(
        ALL_TICKERS,
        progress_cb=lambda i, t, tk: print(f"  [{i+1:3d}/{t}] {tk}", flush=True),
    )

    csv_path = snap_dir / f"{today}_semaforo.csv"
    df_result.to_csv(csv_path, index=False)
    print(f"\n✅ CSV guardado: {csv_path}  ({len(df_result)} tickers)")

    # resumen rápido en consola
    for señal in df_result["Señal"].unique():
        subset = df_result[df_result["Señal"] == señal]
        tks    = ", ".join(subset["Ticker"].tolist())
        print(f"  {señal:30s} → {tks}")

    # ── 2. PNGs de gráfico ───────────────────────────────────────────────
    try:
        build_figure, build_signals, score_signals, get_chart_data = _load_app_functions()

        for tk in CHART_TICKERS:
            print(f"\n  Generando gráfico: {tk}…")
            try:
                data = get_chart_data(tk, period="2y")
                if not data:
                    print(f"  ⚠️  Sin datos para {tk}")
                    continue
                fig      = build_figure(data, tk)
                png_path = snap_dir / f"{today}_{tk.replace('=','').replace('-','_')}.png"
                fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="#0d0f14")
                plt.close(fig)
                print(f"  ✅ PNG guardado: {png_path}")
            except Exception as e:
                print(f"  ❌ Error en {tk}: {e}")
    except Exception as e:
        print(f"\n  ⚠️  No se pudieron generar PNGs: {e}")
        print("      Los CSVs se han guardado correctamente.")

    print(f"\n{'='*60}")
    print(f"  Snapshot completado → {snap_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()