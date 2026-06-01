# indicators.py
# ─────────────────────────────────────────────────────────────────────────────
# Librería compartida: descarga, indicadores, semáforo, dashboard numérico
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from ta.trend  import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator


# ══════════════════════════════════════════════════════════════════════════════
# DESCARGA Y LIMPIEZA
# ══════════════════════════════════════════════════════════════════════════════

def download_df(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    return clean_yf_df(df)


def clean_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()
    return df[needed].dropna().copy()


# ══════════════════════════════════════════════════════════════════════════════
# INDICADORES NÚCLEO
# ══════════════════════════════════════════════════════════════════════════════

def mcginley_dynamic(close: pd.Series, period: int = 25) -> pd.Series:
    k  = 0.6
    md = close.astype(float).copy()
    for i in range(1, len(close)):
        prev = md.iloc[i - 1]
        cur  = close.iloc[i]
        if prev == 0 or pd.isna(prev):
            md.iloc[i] = cur
        else:
            md.iloc[i] = prev + (cur - prev) / (k * period * (cur / prev) ** 4)
    return md


def calculate_pvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    pvi = pd.Series(index=close.index, dtype=float)
    pvi.iloc[0] = 1000.0
    for i in range(1, len(close)):
        if volume.iloc[i] > volume.iloc[i - 1]:
            pct = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]
            pvi.iloc[i] = pvi.iloc[i - 1] * (1 + pct)
        else:
            pvi.iloc[i] = pvi.iloc[i - 1]
    return pvi


def calculate_nvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    nvi = pd.Series(index=close.index, dtype=float)
    nvi.iloc[0] = 1000.0
    for i in range(1, len(close)):
        if volume.iloc[i] < volume.iloc[i - 1]:
            pct = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]
            nvi.iloc[i] = nvi.iloc[i - 1] * (1 + pct)
        else:
            nvi.iloc[i] = nvi.iloc[i - 1]
    return nvi


def calc_mfi_blai5(high, low, close, volume, length: int = 14) -> pd.Series:
    src = (high + low + close) / 3.0
    up  = (volume * np.where(src.diff() > 0, src, 0)).rolling(length).sum()
    dn  = (volume * np.where(src.diff() < 0, src, 0)).rolling(length).sum()
    rs  = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_stoch(src, high, low, length: int = 21, smooth_fast_d: int = 3) -> pd.Series:
    ll = low.rolling(length).min()
    hh = high.rolling(length).max()
    k  = 100 * (src - ll) / (hh - ll)
    return k.rolling(smooth_fast_d).mean()


def awesome_osc(high: pd.Series, low: pd.Series) -> pd.Series:
    mid = (high + low) / 2.0
    return mid.rolling(5).mean() - mid.rolling(34).mean()


# ══════════════════════════════════════════════════════════════════════════════
# BBWP 13 / 252
# ══════════════════════════════════════════════════════════════════════════════

def calculate_bbwp(
    close: pd.Series,
    bb_len: int = 13,
    lookback: int = 252,
) -> tuple[pd.Series, pd.Series]:
    basis = close.rolling(bb_len).mean()
    dev   = close.rolling(bb_len).std(ddof=0)
    bbw   = 2.0 * dev / basis.replace(0, np.nan)

    arr  = bbw.values
    n    = len(arr)
    bbwp = np.full(n, np.nan)

    for i in range(bb_len, n):
        cur = arr[i]
        if np.isnan(cur):
            continue
        start  = max(0, i - lookback)
        window = arr[start:i]
        valid  = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        bbwp[i] = np.sum(valid <= cur) / len(valid) * 100.0

    return pd.Series(bbw, index=close.index), pd.Series(bbwp, index=close.index)


def bbwp_signal(bbwp_pct, bbwp_series):
    if pd.isna(bbwp_pct):
        return "⚪", "normal", "→", "nan%"
    if bbwp_pct < 20:
        punto, zona = "🟢", "compresion"
    elif bbwp_pct > 80:
        punto, zona = "🔴", "expansion"
    else:
        punto, zona = "⚪", "normal"
    reciente = bbwp_series.dropna().iloc[-3:]
    if len(reciente) >= 2:
        slope = reciente.iloc[-1] - reciente.iloc[0]
        pendiente = "↑" if slope > 3 else ("↓" if slope < -3 else "→")
    else:
        pendiente = "→"
    return punto, zona, pendiente, f"{bbwp_pct:.1f}%"


# ══════════════════════════════════════════════════════════════════════════════
# BLAI5 KONCORDE
# ══════════════════════════════════════════════════════════════════════════════

def compute_blai5_koncorde(df: pd.DataFrame, m: int = 15) -> pd.DataFrame:
    df = clean_yf_df(df)
    if df.empty or len(df) < 100:
        return pd.DataFrame()

    ohlc4 = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    pvi   = calculate_pvi(df["Close"], df["Volume"])
    nvi   = calculate_nvi(df["Close"], df["Volume"])
    pvim  = pvi.ewm(span=m, adjust=False).mean()
    nvim  = nvi.ewm(span=m, adjust=False).mean()

    oscp = (pvi - pvim) * 100 / (
        pvim.rolling(90).max() - pvim.rolling(90).min()
    ).replace(0, np.nan)
    azul = (nvi - nvim) * 100 / (
        nvim.rolling(90).max() - nvim.rolling(90).min()
    ).replace(0, np.nan)

    xmf     = calc_mfi_blai5(df["High"], df["Low"], df["Close"], df["Volume"], 14)
    basis   = ohlc4.rolling(25).mean()
    dev     = 2.0 * ohlc4.rolling(25).std()
    bollosc = ((ohlc4 - basis) / dev).replace([np.inf, -np.inf], np.nan) * 100
    xrsi    = RSIIndicator(close=ohlc4, window=14).rsi()
    stoc    = calc_stoch(ohlc4, df["High"], df["Low"], 21, 3)

    marron = (xrsi + xmf + bollosc + stoc / 3.0) / 2.0
    verde  = marron + oscp
    media  = marron.ewm(span=m, adjust=False).mean()

    out = pd.DataFrame(index=df.index)
    out["azul"]   = azul
    out["marron"] = marron
    out["verde"]  = verde
    out["media"]  = media
    return out


def blai5_signals(kdf: pd.DataFrame) -> pd.DataFrame:
    kdf      = kdf.copy()
    valid    = kdf[["verde", "marron", "azul", "media"]].notna().all(axis=1)
    area_max = kdf[["verde", "marron", "azul"]].max(axis=1)
    area_min = kdf[["verde", "marron", "azul"]].min(axis=1)
    inside   = valid & (kdf["media"] >= area_min) & (kdf["media"] <= area_max)

    punto_media, velas_konk = [], []
    estado, conteo = None, 0
    for i in range(len(kdf)):
        if not valid.iloc[i]:
            punto_media.append(False); velas_konk.append(0); continue
        if inside.iloc[i]:
            if estado != "inside": estado, conteo = "inside", 1
            else: conteo += 1
            punto_media.append(True)
        else:
            if estado != "outside": estado, conteo = "outside", 1
            else: conteo += 1
            punto_media.append(False)
        velas_konk.append(conteo)

    kdf["punto_media_verde"] = punto_media
    kdf["velas_konk"]        = velas_konk
    return kdf


# ══════════════════════════════════════════════════════════════════════════════
# BITMAN
# ══════════════════════════════════════════════════════════════════════════════

def clasificar_bitman(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or len(df) < 60:
        return pd.DataFrame()

    out     = df.copy()
    adx_ind = ADXIndicator(high=out["High"], low=out["Low"], close=out["Close"], window=14)
    out["ADX"]       = adx_ind.adx()
    out["ADX_Slope"] = out["ADX"].diff().rolling(3).mean()
    out["AO"]        = awesome_osc(out["High"], out["Low"])
    diff             = out["AO"] - out["AO"].shift(1)
    out["AO_Color"]  = np.where(diff <= 0, "rojo", "verde")

    slope_mean_abs = out["ADX_Slope"].abs().rolling(20).mean()
    weak_thr       = (slope_mean_abs * 0.25).fillna(np.nan)

    out["ADX_Giro"]        = False
    out["Bitman_Color"]    = out["AO_Color"]
    out["Bitman_Etiqueta"] = "INDEFINICIÓN"
    out["Bitman_Velas"]    = 0

    last_turn_idx = None
    current_color = "verde"
    counter       = 0

    for i in range(len(out)):
        adx_slope_now = out["ADX_Slope"].iloc[i]
        ao_color_now  = out["AO_Color"].iloc[i]

        if (pd.isna(adx_slope_now) or pd.isna(weak_thr.iloc[i]) or
                abs(adx_slope_now) <= weak_thr.iloc[i]):
            counter += 1
            out.iloc[i, out.columns.get_loc("Bitman_Etiqueta")] = "INDEFINICIÓN"
            out.iloc[i, out.columns.get_loc("Bitman_Color")]    = ao_color_now
            out.iloc[i, out.columns.get_loc("Bitman_Velas")]    = counter
            continue

        adx_dir    = "impulso" if adx_slope_now > 0 else "retroceso"
        prev_slope = out["ADX_Slope"].iloc[i - 1] if i > 0 else np.nan
        prev_weak  = weak_thr.iloc[i - 1]          if i > 0 else np.nan
        giro = (i > 0
                and not pd.isna(prev_slope)
                and np.sign(prev_slope) != np.sign(adx_slope_now)
                and (pd.isna(prev_weak) or abs(prev_slope) > prev_weak))

        if giro:
            out.iloc[i, out.columns.get_loc("ADX_Giro")] = True
            last_turn_idx = i
            counter       = 1
            ao_w    = out["AO_Color"].iloc[max(0, i - 4): i + 1]
            changes = ao_w[ao_w != ao_w.shift(1)].dropna()
            if len(changes) >= 1:
                current_color = changes.iloc[-1]
        else:
            if last_turn_idx is not None and 0 < (i - last_turn_idx) <= 4:
                ao_w    = out["AO_Color"].iloc[last_turn_idx: i + 1]
                changes = ao_w[ao_w != ao_w.shift(1)].dropna()
                if len(changes) > 0:
                    current_color = changes.iloc[-1]
            counter = (i - last_turn_idx + 1) if last_turn_idx is not None else counter + 1

        etiqueta = (
            "IMPULSO ALCISTA"   if adx_dir == "impulso"   and current_color == "verde" else
            "IMPULSO BAJISTA"   if adx_dir == "impulso"   and current_color == "rojo"  else
            "RETROCESO ALCISTA" if adx_dir == "retroceso" and current_color == "verde" else
            "RETROCESO BAJISTA"
        )
        out.iloc[i, out.columns.get_loc("Bitman_Color")]    = current_color
        out.iloc[i, out.columns.get_loc("Bitman_Etiqueta")] = etiqueta
        out.iloc[i, out.columns.get_loc("Bitman_Velas")]    = counter

    return out


# ══════════════════════════════════════════════════════════════════════════════
# DIVERGENCIAS RSI
# ══════════════════════════════════════════════════════════════════════════════

def detectar_divergencia_simple(
    df: pd.DataFrame,
    lookback: int = 80,
    order: int = 3,
    max_gap: int = 20,
    tol: float = 0.005,
) -> pd.DataFrame:
    close = df["Close"].copy()
    rsi   = RSIIndicator(close=close, window=14).rsi()
    p, o  = close.values, rsi.values
    n     = len(p)

    def pivots(vals, kind="low"):
        idxs = []
        for i in range(order, n - order):
            if np.isnan(vals[i]):
                continue
            w = vals[i - order: i + order + 1]
            if kind == "low":
                if vals[i] <= np.nanmin(w) * (1 + tol) and vals[i] <= vals[i-1] and vals[i] <= vals[i+1]:
                    idxs.append(i)
            else:
                if vals[i] >= np.nanmax(w) * (1 - tol) and vals[i] >= vals[i-1] and vals[i] >= vals[i+1]:
                    idxs.append(i)
        return idxs

    def nearest(ref, cands):
        best, best_d = None, 10**9
        for c in cands:
            d = abs(c - ref)
            if d <= max_gap and d < best_d:
                best, best_d = c, d
        return best

    pl = pivots(p, "low");  ph = pivots(p, "high")
    ol = pivots(o, "low");  oh = pivots(o, "high")
    alc_idx = baj_idx = None

    for j in range(1, len(pl)):
        p1, p2 = pl[j-1], pl[j]
        if p2 - p1 > lookback: continue
        i1, i2 = nearest(p1, ol), nearest(p2, ol)
        if i1 is None or i2 is None or i1 == i2: continue
        if p[p2] < p[p1] and o[i2] > o[i1]:
            alc_idx = p2

    for j in range(1, len(ph)):
        p1, p2 = ph[j-1], ph[j]
        if p2 - p1 > lookback: continue
        i1, i2 = nearest(p1, oh), nearest(p2, oh)
        if i1 is None or i2 is None or i1 == i2: continue
        if p[p2] > p[p1] and o[i2] < o[i1]:
            baj_idx = p2

    if alc_idx is not None and baj_idx is not None:
        div_tipo, div_idx = ("alcista", alc_idx) if alc_idx >= baj_idx else ("bajista", baj_idx)
    elif alc_idx is not None:
        div_tipo, div_idx = "alcista", alc_idx
    elif baj_idx is not None:
        div_tipo, div_idx = "bajista", baj_idx
    else:
        div_tipo, div_idx = "ninguna", None

    out = df.copy()
    out["divergencia_tipo"] = "ninguna"
    out["divergencia"]      = "⚪"
    if div_idx is not None:
        out.iloc[div_idx, out.columns.get_loc("divergencia_tipo")] = div_tipo
        out.iloc[div_idx, out.columns.get_loc("divergencia")]      = (
            "🟢" if div_tipo == "alcista" else "🔴"
        )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS SEMÁFORO
# ══════════════════════════════════════════════════════════════════════════════

def velas_desde_activacion(serie_bool: pd.Series) -> int:
    vals = serie_bool.fillna(False).values
    n    = len(vals)
    if n == 0 or not vals[-1]:
        return 999
    count = 0
    for i in range(n - 1, -1, -1):
        if vals[i]: count += 1
        else:       break
    return count


def azul_z_score(kdf: pd.DataFrame, window: int = 60) -> float:
    azul = kdf["azul"].dropna()
    if len(azul) < window + 4:
        return 0.0
    slope = azul.iloc[-1] - azul.iloc[-4]
    std   = azul.rolling(window).std().iloc[-1]
    if pd.isna(std) or std == 0:
        return 0.0
    return slope / std


def calcular_velas_señal(close, volume, kdf, macd_line, macd_signal_line, bitman_df) -> dict:
    gap         = macd_line - macd_signal_line
    macd_activo = (gap > 0) & (gap.diff() > 0)
    v_macd      = velas_desde_activacion(macd_activo)

    azul        = kdf["azul"].fillna(0)
    azul_activo = (azul > 0) & ((azul - azul.shift(3).fillna(0)) > 0)
    v_azul      = velas_desde_activacion(azul_activo)

    v_media = velas_desde_activacion(pd.Series(kdf["punto_media_verde"].values, index=kdf.index))

    pvi_s    = calculate_pvi(close, volume)
    pvi_ema  = pvi_s.ewm(span=25, adjust=False).mean()
    pvi_gap  = pvi_s - pvi_ema
    v_pvi    = velas_desde_activacion((pvi_s > pvi_ema) & (pvi_gap.diff() > 0))

    _, bbwp_s = calculate_bbwp(close, bb_len=13, lookback=252)
    indices   = np.where(bbwp_s.fillna(False).values < 20)[0]
    v_bbwp_comp = (len(bbwp_s) - 1 - indices[-1]) if len(indices) > 0 else 999

    if bitman_df is not None and not bitman_df.empty:
        v_bitman = velas_desde_activacion(
            pd.Series((bitman_df["Bitman_Etiqueta"] == "IMPULSO ALCISTA").values,
                      index=bitman_df.index)
        )
    else:
        v_bitman = 999

    return dict(v_macd=v_macd, v_azul=v_azul, v_media=v_media,
                v_pvi=v_pvi, v_bbwp_comp=v_bbwp_comp, v_bitman=v_bitman)


# ══════════════════════════════════════════════════════════════════════════════
# SEMÁFORO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def semaforo(data: dict, velas: dict) -> dict:
    gap, accel = data["macd_gap"], data["macd_accel"]
    if gap >= 0 and accel > 0:
        c1, c1_txt = "🟢", f"MACD 🟢 acelerando ({velas['v_macd']}v)"
    elif gap >= 0:
        c1, c1_txt = "⚪", "MACD ⚪ decelerando"
    else:
        c1, c1_txt = "🔴", "MACD 🔴 negativo"

    azul_verde, azul_slope = data["konk_azul_verde"], data.get("azul_slope", 0.0)
    if azul_verde and azul_slope > 0:
        c2, c2_txt = "🟢", f"Azul K 🟢 positivo↑ ({velas['v_azul']}v)"
    elif azul_verde:
        c2, c2_txt = "⚪", "Azul K ⚪ positivo plano"
    else:
        c2, c2_txt = "🔴", "Azul K 🔴 negativo"

    if data["konk_punto_verde"]:
        c3, c3_txt = "🟢", f"Media K 🟢 en área ({velas['v_media']}v)"
    else:
        c3, c3_txt = "🔴", "Media K 🔴 fuera"

    bbwp_pct, bbwp_pend = data.get("bbwp_pct", 50.0), data.get("bbwp_pendiente", "→")
    v_bbwp_comp         = velas["v_bbwp_comp"]
    bbwp_comp_txt       = f" (compresión hace {v_bbwp_comp}v)" if v_bbwp_comp < 40 else ""
    if bbwp_pct > 60:
        c4, c4_txt = "🟢", f"BBWP 🟢 alto {bbwp_pct:.0f}%{bbwp_comp_txt}"
    elif bbwp_pct < 40 and bbwp_pend == "↑":
        c4, c4_txt = "🟢", f"BBWP 🟢 cargando {bbwp_pct:.0f}%↑"
    elif bbwp_pct < 40:
        c4, c4_txt = "🔴", f"BBWP 🔴 compresión plana {bbwp_pct:.0f}%"
    else:
        c4, c4_txt = "⚪", f"BBWP ⚪ zona media {bbwp_pct:.0f}%"

    pvi_activo, pvi_accel = data.get("pvi_activo", False), data.get("pvi_accel", False)
    if pvi_activo and pvi_accel:
        c8, c8_txt = "🟢", f"PVI 🟢 sobre EMA25 y acelerando ({velas['v_pvi']}v)"
    elif pvi_activo:
        c8, c8_txt = "⚪", "PVI ⚪ sobre EMA25 decelerando"
    else:
        c8, c8_txt = "🔴", "PVI 🔴 bajo EMA25"

    precio, e200 = data.get("precio", 0), data.get("e200", 0)
    if data.get("cerca_mcg25") or data.get("cerca_e200"):
        c5, c5_txt = "🟠", "🟠 precio en zona soporte MCG25/EMA200"
    else:
        c5     = "⚪" if precio > e200 else "🔴"
        c5_txt = "sobre EMA200" if precio > e200 else "🔴 bajo EMA200 — precaución"

    b_etiq  = data.get("bitman_etiqueta", "")
    b_velas = data.get("bitman_velas", 999)
    v_bf    = velas["v_bitman"]
    if b_etiq == "IMPULSO ALCISTA":
        c6, c6_txt = "🟢", f"Bitman 🟢 impulso alcista (ciclo {b_velas}v / fresco {v_bf}v)"
    elif b_etiq == "RETROCESO ALCISTA":
        c6, c6_txt = "⚪", f"Bitman ⚪ retroceso alcista ({b_velas}v)"
    elif "BAJISTA" in b_etiq:
        c6, c6_txt = "🔴", f"Bitman 🔴 {b_etiq.lower()} ({b_velas}v)"
    else:
        c6, c6_txt = "⚪", f"Bitman ⚪ indefinición ({b_velas}v)"

    azul_z = data.get("azul_z", 0.0)
    c7     = "⚡" if azul_z > 1.5 else "⚪"
    c7_txt = f"⚡ Azul pendiente fuerte (z={azul_z:.1f})" if c7 == "⚡" else ""

    verde_val = data.get("konk_verde_val", 0.0)
    azul_val  = data.get("konk_azul_val",  0.0)
    atenc_konk = (verde_val < 0) and (azul_val > 0)

    div       = data.get("divergencia_tipo",  "ninguna")
    div_velas = data.get("divergencia_velas", 999)

    n_activas = sum([c1 == "🟢", c2 == "🟢", c3 == "🟢", c4 == "🟢", c8 == "🟢"])
    n_rojas   = sum([c1 == "🔴", c2 == "🔴", c3 == "🔴", c4 == "🔴", c8 == "🔴"])

    confluencia_fresca = (velas["v_macd"] <= 5 and velas["v_media"] <= 5
                          and c1 == "🟢" and c3 == "🟢")
    señal_tardia  = (n_activas >= 4 and not confluencia_fresca)
    inicio_desact = (n_activas >= 3 and n_rojas >= 1 and (c1 == "🔴" or c3 == "🔴"))
    mayoria_desact = n_rojas >= 3

    if atenc_konk and n_activas < 4:
        decision, score_str = "⚠️ ATENCIÓN KONKORDE", "K!"
    elif n_activas == 5 and confluencia_fresca and c6 == "🟢":
        decision, score_str = "🚀 COMPRA 100%", "5/5 + B"
    elif n_activas >= 4 and confluencia_fresca:
        decision, score_str = "🟡 COMPRA 50%", f"{n_activas}/5"
    elif n_activas >= 4 and señal_tardia:
        decision, score_str = "⏰ LLEGAS TARDE", f"tarde {n_activas}/5"
    elif inicio_desact:
        decision, score_str = "⚠️ VIGILAR SALIDA", f"sal {n_activas}/5"
    elif mayoria_desact:
        decision, score_str = "🔴 VENTA", f"{n_activas}/5"
    elif n_activas == 3:
        decision, score_str = "👀 VIGILAR", "3/5"
    elif n_activas <= 1 and n_rojas >= 3:
        decision, score_str = "⛔ NI DE COÑA", f"{n_activas}/5"
    else:
        decision, score_str = "⛔ SIN SETUP", f"{n_activas}/5"

    razones = [c1_txt, c2_txt, c3_txt, c4_txt, c8_txt, c5_txt, c6_txt]
    if c7 == "⚡":  razones.append(c7_txt)
    if velas["v_bbwp_comp"] < 30:
        razones.append(f"BBWP tuvo compresión hace {velas['v_bbwp_comp']}v — energía acumulada")
    if div == "alcista":
        if div_velas <= 5:   razones.append(f"🟢 Div alcista RSI FRESCA ({div_velas}v)")
        elif div_velas <= 20: razones.append(f"🟢 Div alcista RSI válida ({div_velas}v)")
        elif div_velas <= 50: razones.append(f"Div alcista RSI contexto ({div_velas}v)")
    elif div == "bajista" and div_velas <= 20:
        razones.append(f"🔴 Div bajista RSI ({div_velas}v) — cautela")
    if precio < e200:
        razones.append("🔴 Precio bajo EMA200 — contexto bajista")
    if atenc_konk:
        razones.append("⚠️ Verde K negativa + Azul K positivo — señal independiente potente")

    return dict(
        decision=decision, score=score_str,
        c1=c1, c2=c2, c3=c3, c4=c4, c8=c8, c5=c5, c6=c6, c7=c7,
        atencion_konk=atenc_konk,
        razones=" | ".join(r for r in razones if r),
    )


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD TEXTO  (Bloque 1)
# ══════════════════════════════════════════════════════════════════════════════

def get_sovereign_dashboard(tickers: list, progress_cb=None) -> pd.DataFrame:
    """
    Devuelve el DataFrame del semáforo para todos los tickers.
    progress_cb: callable(i, total, ticker) para progresión en Streamlit.
    """
    report = []
    total  = len(tickers)

    for idx_t, t in enumerate(tickers):
        if progress_cb:
            progress_cb(idx_t, total, t)
        try:
            df = download_df(t, period="2y", interval="1d")
            if df.empty or len(df) < 150:
                continue

            close, volume = df["Close"], df["Volume"]
            precio        = close.iloc[-1]

            mcg25     = mcginley_dynamic(close, 25)
            e200_s    = EMAIndicator(close=close, window=200).ema_indicator()
            mcg25_val = mcg25.iloc[-1]
            e200_val  = e200_s.iloc[-1]

            cerca_mcg  = mcg25_val * 0.988 <= precio <= mcg25_val * 1.012
            cerca_e200 = e200_val  * 0.985 <= precio <= e200_val  * 1.015
            s_mcg  = "🟡" if cerca_mcg  else ("🟢" if precio > mcg25_val else "🔴")
            s_e200 = "🟡" if cerca_e200 else ("🟢" if precio > e200_val  else "🔴")
            trend_str = f"MD25:{s_mcg} E200:{s_e200}"

            rsi       = RSIIndicator(close=close).rsi()
            cruce_30  = (rsi > 30) & (rsi.shift(1) <= 30)
            cruce_70  = (rsi < 70) & (rsi.shift(1) >= 70)
            is_active, velas_rsi = False, 0
            for i in range(len(rsi)):
                if cruce_30.iloc[i]:
                    is_active = True; velas_rsi = 1
                elif is_active:
                    if cruce_70.iloc[i] or velas_rsi >= 10:
                        is_active = False; velas_rsi = 0
                    else:
                        velas_rsi += 1
            rsi_str = f"{'🟢' if is_active else '🔴'} {velas_rsi}v {'➕' if rsi.iloc[-1] > 50 else '➖'}"

            macd_obj         = MACD(close=close)
            macd_line        = macd_obj.macd()
            macd_signal_line = macd_obj.macd_signal()
            macd_diff        = macd_obj.macd_diff()
            gap              = macd_line - macd_signal_line
            gap_vol          = gap.abs().rolling(20).mean()
            accel            = (gap.diff() / gap_vol).fillna(0).iloc[-1]
            macd_gap_v       = gap.iloc[-1]
            estado_macd      = "Sep" if (gap.iloc[-1] * accel) >= 0 else "Jun"
            macd_str = (
                f"{'🟢' if macd_diff.iloc[-1] > 0 else '🔴'} "
                f"{'➕' if macd_line.iloc[-1] >= 0 else '➖'} "
                f"{accel:.2f} {estado_macd}"
            )

            kdf = compute_blai5_koncorde(df, m=15)
            if kdf.empty:
                continue
            kdf = blai5_signals(kdf)

            azul_verde    = kdf["azul"].iloc[-1] > 0
            azul_val_now  = float(kdf["azul"].iloc[-1])
            verde_val_now = float(kdf["verde"].iloc[-1])
            punto_verde   = kdf["punto_media_verde"].iloc[-1]
            velas_konk_v  = int(kdf["velas_konk"].iloc[-1])
            konk_str = (
                f"{'🟢' if azul_verde else '🔴'}{'🟢' if punto_verde else '🔴'}"
                f" {velas_konk_v}v"
            )

            azul_z_val   = azul_z_score(kdf)
            azul_slope_v = (1.0 if (len(kdf["azul"].dropna()) >= 4 and
                                    kdf["azul"].iloc[-1] - kdf["azul"].iloc[-4] > 0) else -1.0)

            bitman_df  = clasificar_bitman(df)
            if bitman_df.empty:
                continue
            bitman_row     = bitman_df.iloc[-1]
            bitman_etiq    = bitman_row["Bitman_Etiqueta"]
            bitman_velas_v = int(bitman_row["Bitman_Velas"])
            bitman_alcista = bitman_etiq in ("IMPULSO ALCISTA", "RETROCESO ALCISTA")
            emoji_bitman   = "📈" if bitman_alcista else ("📉" if "BAJISTA" in bitman_etiq else "⬜")

            div_df  = detectar_divergencia_simple(df)
            hits    = div_df[div_df["divergencia_tipo"] != "ninguna"]
            if not hits.empty:
                div_tipo  = hits.iloc[-1]["divergencia_tipo"]
                div_idx   = div_df.index.get_loc(hits.index[-1])
                div_velas = len(div_df) - 1 - div_idx
            else:
                div_tipo, div_velas = "ninguna", 999

            if div_tipo != "ninguna":
                e, ec = ("🟢", "🟡") if div_tipo == "alcista" else ("🔴", "🟠")
                if div_velas <= 5:    div_str = f"{e} {div_tipo.upper()} FRESCA ({div_velas}v)"
                elif div_velas <= 20: div_str = f"{e} {div_tipo.upper()} válida ({div_velas}v)"
                elif div_velas <= 50: div_str = f"{ec} {div_tipo.upper()} ctx ({div_velas}v)"
                else:                 div_str = f"⚪ {div_tipo} caduc ({div_velas}v)"
            else:
                div_str = "⚪"

            _, bbwp_s = calculate_bbwp(close, bb_len=13, lookback=252)
            bbwp_last = bbwp_s.iloc[-1]
            punto_bbwp, bbwp_zona, pend_bbwp, nivel_bbwp = bbwp_signal(bbwp_last, bbwp_s)
            bbwp_str = f"{punto_bbwp} {pend_bbwp} {nivel_bbwp}"

            pvi_s      = calculate_pvi(close, volume)
            pvi_ema    = pvi_s.ewm(span=25, adjust=False).mean()
            pvi_gap    = pvi_s - pvi_ema
            pvi_activo = bool(pvi_s.iloc[-1] > pvi_ema.iloc[-1])
            pvi_accel  = bool(pvi_gap.diff().iloc[-1] > 0)

            velas_señal = calcular_velas_señal(
                close, volume, kdf, macd_line, macd_signal_line, bitman_df
            )

            def fv(v): return f"{v}v" if v < 999 else "—"

            pvi_str    = f"{'🟢' if (pvi_activo and pvi_accel) else ('⚪' if pvi_activo else '🔴')} {fv(velas_señal['v_pvi'])}"
            velas_str  = (f"M:{fv(velas_señal['v_macd'])} Az:{fv(velas_señal['v_azul'])} "
                          f"Me:{fv(velas_señal['v_media'])} B:{fv(velas_señal['v_bitman'])}")
            bitman_str = (f"{bitman_etiq} (ciclo {bitman_velas_v}v / fresco {fv(velas_señal['v_bitman'])}) "
                          f"{emoji_bitman}")

            input_data = dict(
                precio=precio, e200=e200_val,
                cerca_mcg25=cerca_mcg, cerca_e200=cerca_e200,
                bitman_etiqueta=bitman_etiq, bitman_velas=bitman_velas_v,
                konk_azul_verde=azul_verde, konk_azul_val=azul_val_now,
                konk_verde_val=verde_val_now, konk_punto_verde=punto_verde,
                konk_velas=velas_konk_v, azul_z=azul_z_val, azul_slope=azul_slope_v,
                macd_gap=macd_gap_v, macd_accel=accel,
                pvi_activo=pvi_activo, pvi_accel=pvi_accel,
                divergencia_tipo=div_tipo, divergencia_velas=div_velas,
                bbwp_pct=float(bbwp_last) if not pd.isna(bbwp_last) else 50.0,
                bbwp_zona=bbwp_zona, bbwp_pendiente=pend_bbwp,
            )
            analisis = semaforo(input_data, velas_señal)

            report.append(dict(
                Ticker=t, Tendencia=trend_str, RSI=rsi_str, MACD=macd_str,
                Koncorde=konk_str, PVI=pvi_str, Bitman=bitman_str,
                Div=div_str, BBWP=bbwp_str,
                **{"Velas⏱": velas_str},
                Score=analisis["score"], Señal=analisis["decision"],
                Razones=analisis["razones"],
            ))

        except Exception as e:
            print(f"❌ {t}: {e}")
            continue

    result = pd.DataFrame(report)
    if not result.empty:
        orden = {
            "⚠️ ATENCIÓN KONKORDE": 0, "🚀 COMPRA 100%": 1, "🟡 COMPRA 50%": 2,
            "👀 VIGILAR": 3, "⏰ LLEGAS TARDE": 4, "⚠️ VIGILAR SALIDA": 5,
            "🔴 VENTA": 6, "⛔ SIN SETUP": 7, "⛔ NI DE COÑA": 8,
        }
        result["_sort"] = result["Señal"].map(orden).fillna(9)
        result = (result.sort_values(["_sort", "Ticker"])
                        .drop(columns="_sort")
                        .reset_index(drop=True))
    return result