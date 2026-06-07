# lcrack_sniper.py
# ─────────────────────────────────────────────────────────────────────────────
# Motor LCrack V6.2 Sniper - Separado para modularidad
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def calcular_motor_v62_sniper(df: pd.DataFrame) -> pd.DataFrame:
    """
    Motor matemático LCrack V6.2 calibrado para máxima eficiencia.
    Retorna el DataFrame con los estados e inercias calculadas.
    """
    if df is None or df.empty or len(df) < 120:
        if df is not None:
            df["Sniper_State"] = "IGNORAR"
            df["Sniper_Detalle"] = "Historial insuficiente (< 120 velas)"
            df["Sniper_Velas_Activacion"] = 999
        return df

    close_vals = df['Close'].values
    w_long = 120
    w_short = 10
    
    # 1. ATR Estructural (14)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    df['ATR_Sniper'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(window=14).mean()
    
    # 2. BBWP Institucional (13/252)
    mid = df['Close'].rolling(window=13).mean()
    std = df['Close'].rolling(window=13).std()
    bbw = (std * 4) / mid if mid is not None else (df['High'] - df['Low']) / df['Close']
    df['BBWP_Sniper'] = bbw.rolling(window=252).apply(
        lambda x: (x[-1] >= x).sum() / len(x) if len(x) > 0 else 0.0, raw=True
    )

    # 3. Regresiones Lineales Normalizadas
    slopes_long = np.zeros(len(df))
    reg_long_vals = np.zeros(len(df))
    slopes_short = np.zeros(len(df))

    for i in range(w_long, len(df)):
        slice_past = close_vals[:i+1]
        
        # Inercia Macro (120 velas)
        y_l = slice_past[-w_long:]
        m_l, b_l = np.polyfit(np.arange(w_long), y_l, 1)
        mean_l = np.mean(y_l)
        slopes_long[i] = m_l / mean_l if mean_l != 0 else 0.0
        reg_long_vals[i] = m_l * (w_long - 1) + b_l
        
        # Gatillo Corto (10 velas)
        y_s = slice_past[-w_short:]
        m_s, _ = np.polyfit(np.arange(w_short), y_s, 1)
        mean_s = np.mean(y_s)
        slopes_short[i] = m_s / mean_s if mean_s != 0 else 0.0

    df['slope_long_sn'] = slopes_long
    df['reg_long_sn'] = reg_long_vals
    df['slope_short_sn'] = slopes_short
    df['slope_short_prev_sn'] = df['slope_short_sn'].shift(1)

    states = []
    razones_lista = []
    velas_activacion = []
    ticks_since_setup = 999

    for idx in range(len(df)):
        if idx < w_long:
            states.append("IGNORAR")
            razones_lista.append("Historial insuficiente")
            velas_activacion.append(999)
            continue

        current_price = df['Close'].iloc[idx]
        reg_l = df['reg_long_sn'].iloc[idx]
        atr = df['ATR_Sniper'].iloc[idx]
        s_long = df['slope_long_sn'].iloc[idx]
        s_short = df['slope_short_sn'].iloc[idx]
        s_short_prev = df['slope_short_prev_sn'].iloc[idx]
        bbwp_val = df['BBWP_Sniper'].iloc[idx]

        # Umbrales estrictos V6.2 (Techo Matemático)
        cond_trend = s_long > 0.00045                  
        cond_value = current_price <= (reg_l + (0.10 * atr)) 

        if cond_trend and cond_value:
            ticks_since_setup = 0
        else:
            ticks_since_setup += 1

        cond_trigger = (s_short > 0) and (s_short_prev <= 0)
        cond_volatilidad = (bbwp_val >= 0.15) and (bbwp_val <= 0.40) 

        if cond_trigger and (ticks_since_setup <= 1) and cond_volatilidad:
            states.append("ULTRA-SAFE SNIPER")
            razones_lista.append(f"Sniper V6.2 OK | Inercia: {s_long:.5f} | BBWP: {bbwp_val:.2f}")
            velas_activacion.append(0)
            continue
        
        if s_long > 0.00010 and current_price <= reg_l:
            states.append("MONITOR")
            razones_lista.append(f"Inercia alcista ({s_long:.5f}). Esperando compresión/gatillo en suelo.")
        else:
            states.append("IGNORAR")
            if s_long <= 0:
                razones_lista.append("Sin inercia estructural (Tendencia bajista o lateral muerta).")
            elif bbwp_val < 0.15:
                razones_lista.append(f"Volatilidad en rango muerto (BBWP: {bbwp_val:.2f}).")
            else:
                razones_lista.append("Precio extendido fuera de parámetros operativos.")
        velas_activacion.append(ticks_since_setup + 1)

    df["Sniper_State"] = states
    df["Sniper_Detalle"] = razones_lista
    df["Sniper_Velas_Activacion"] = velas_activacion
    return df
