# analisis_sniper_pepino.py
import pandas as pd
import numpy as np
from app import ALL_TICKERS
from indicators import download_df, get_sovereign_dashboard_v2, calcular_motor_v62_sniper

def analizar_correlacion_sniper_pepino(tickers: list, periodo: str = "2y"):
    """
    Analiza qué % de señales Pepino coinciden con Sniper activado
    """
    resultados = []
    
    for ticker in tickers[:20]:  # Empezar con 20
        try:
            df = download_df(ticker, periodo, "1d")
            if df.empty or len(df) < 200:
                continue
            
            # Calcular Sniper histórico
            df_sniper = calcular_motor_v62_sniper(df.copy())
            
            # Calcular señales Pepino históricas (necesitas versión que guarde histórico)
            # Por ahora, usamos el dashboard actual que da señal actual
            
            # Contar estados Sniper
            counts = df_sniper["Sniper_State"].value_counts()
            pct_ultra = counts.get("ULTRA-SAFE SNIPER", 0) / len(df_sniper) * 100
            pct_monitor = counts.get("MONITOR", 0) / len(df_sniper) * 100
            
            resultados.append({
                "Ticker": ticker,
                "Dias_ULTRA": counts.get("ULTRA-SAFE SNIPER", 0),
                "Dias_MONITOR": counts.get("MONITOR", 0),
                "Dias_IGNORAR": counts.get("IGNORAR", 0),
                "Pct_ULTRA": pct_ultra,
                "Pct_MONITOR": pct_monitor,
                "Pct_IGNORAR": 100 - pct_ultra - pct_monitor
            })
            
        except Exception as e:
            continue
    
    df_res = pd.DataFrame(resultados)
    
    print("\n" + "="*70)
    print("📊 DISTRIBUCIÓN DE ESTADOS SNIPER POR TICKER")
    print("="*70 + "\n")
    print(df_res.to_string(index=False))
    
    print(f"\n📈 Promedios globales:")
    print(f"  ULTRA-SAFE: {df_res['Pct_ULTRA'].mean():.1f}%")
    print(f"  MONITOR: {df_res['Pct_MONITOR'].mean():.1f}%")
    print(f"  IGNORAR: {df_res['Pct_IGNORAR'].mean():.1f}%")
    
    return df_res

# Ejecutar
df_sniper_stats = analizar_correlacion_sniper_pepino(ALL_TICKERS)
