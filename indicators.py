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



# ══════════════════════════════════════════════════════════════════════════════
# SEÑALES V2 — nuevo sistema de scoring con frescura
# ══════════════════════════════════════════════════════════════════════════════

def _velas_desde_cruce(serie_bool: pd.Series) -> int:
    """
    Devuelve cuántas velas han pasado desde el ÚLTIMO cruce True.
    Si nunca cruzó o no está activo ahora → 999.
    """
    vals = serie_bool.fillna(False).values
    n    = len(vals)
    if n == 0 or not vals[-1]:
        return 999
    # buscar hacia atrás el primer False antes del bloque True final
    for i in range(n - 1, -1, -1):
        if not vals[i]:
            return (n - 1) - i
    return n  # toda la serie es True


def calcular_señales_v2(
    df:        pd.DataFrame,
    kdf:       pd.DataFrame,
    bitman_df: pd.DataFrame,
    bbwp_s:    pd.Series,
    macd_line: pd.Series,
    macd_sig:  pd.Series,
    pvi_s:     pd.Series,
    pvi_ema:   pd.Series,
    rsi_s:     pd.Series,
) -> dict:
    """
    Nuevo sistema de señales con distinción estado / frescura.

    Señales POTENTES (máx 6, con frescura ≤3 velas):
      S1  MACD cruce alcista
      S2  AO cambia rojo→verde
      S3  PVI cruza EMA25 hacia arriba
      S4  Azul Koncorde cruza cero hacia arriba
      S5  Media Koncorde entra en área
      S6  Bitman cambia a IMPULSO ALCISTA

    Señales INFORMATIVAS (máx 2, sin frescura):
      S7  BBWP pendiente positiva y > 20
      S8  Volumen confirmatorio > 1.3x MA20 con cierre positivo

    Señales INFORMATIVAS EXTRA (texto, sin puntuación):
      SI1 Precio en entorno MCG25 ±1.2%
      SI2 Precio en entorno EMA200 ±1.5%
      SI3 Azul+ y Verde- (Atención Konkorde)
      SI4 Azul subiendo (slope positivo)
      SI5 Divergencia RSI alcista ≤15v

    Retorna dict con todo el detalle para mostrar en dashboard.
    """
    UMBRAL_FRESCURA = 3   # velas
    resultado = {
        "señales": {},      # detalle de cada señal
        "informativas": [], # textos informativos
        "n_activas": 0,
        "n_frescas": 0,
        "etiqueta": "⛔ SIN SETUP",
        "razones": "",
    }

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    n      = len(df)

    if n < 60:
        return resultado

    # ── S1: MACD cruce alcista ────────────────────────────────────────────
    diff         = macd_line - macd_sig
    macd_activo  = diff.iloc[-1] > 0
    macd_cruce   = (diff > 0) & (diff.shift(1) <= 0)
    macd_frescura = _velas_desde_cruce(macd_cruce) if macd_activo else 999
    macd_fresca  = macd_activo and macd_frescura <= UMBRAL_FRESCURA
    resultado["señales"]["S1_MACD"] = {
        "nombre":  "MACD cruce ↑",
        "activa":  macd_activo,
        "fresca":  macd_fresca,
        "velas":   macd_frescura if macd_activo else None,
    }

    # ── S2: AO cambia rojo → verde ────────────────────────────────────────
    ao_s        = awesome_osc(high, low)
    ao_verde    = ao_s > ao_s.shift(1)
    ao_activo   = bool(ao_verde.iloc[-1])
    ao_cruce    = ao_verde & ~ao_verde.shift(1).fillna(False)
    ao_frescura = _velas_desde_cruce(ao_cruce) if ao_activo else 999
    ao_fresca   = ao_activo and ao_frescura <= UMBRAL_FRESCURA
    resultado["señales"]["S2_AO"] = {
        "nombre": "AO rojo→verde",
        "activa": ao_activo,
        "fresca": ao_fresca,
        "velas":  ao_frescura if ao_activo else None,
    }

    # ── S3: PVI cruza EMA25 hacia arriba ──────────────────────────────────
    pvi_activo   = bool(pvi_s.iloc[-1] > pvi_ema.iloc[-1])
    pvi_cruce    = (pvi_s > pvi_ema) & (pvi_s.shift(1) <= pvi_ema.shift(1))
    pvi_frescura = _velas_desde_cruce(pvi_cruce) if pvi_activo else 999
    pvi_fresca   = pvi_activo and pvi_frescura <= UMBRAL_FRESCURA
    resultado["señales"]["S3_PVI"] = {
        "nombre": "PVI cruza EMA25 ↑",
        "activa": pvi_activo,
        "fresca": pvi_fresca,
        "velas":  pvi_frescura if pvi_activo else None,
    }

    # ── S4: Azul Koncorde cruza cero hacia arriba ─────────────────────────
    if not kdf.empty and "azul" in kdf.columns:
        azul          = kdf["azul"]
        azul_activo   = bool(azul.iloc[-1] > 0)
        azul_cruce    = (azul > 0) & (azul.shift(1) <= 0)
        azul_frescura = _velas_desde_cruce(azul_cruce) if azul_activo else 999
        azul_fresca   = azul_activo and azul_frescura <= UMBRAL_FRESCURA
    else:
        azul_activo = azul_fresca = False
        azul_frescura = 999
    resultado["señales"]["S4_AZUL"] = {
        "nombre": "Azul K cruza 0 ↑",
        "activa": azul_activo,
        "fresca": azul_fresca,
        "velas":  azul_frescura if azul_activo else None,
    }

    # ── S5: Media Koncorde entra en área ──────────────────────────────────
    if not kdf.empty and all(c in kdf.columns for c in ["verde","marron","azul","media"]):
        area_max    = kdf[["verde","marron","azul"]].max(axis=1)
        area_min    = kdf[["verde","marron","azul"]].min(axis=1)
        media       = kdf["media"]
        en_area     = (media >= area_min) & (media <= area_max) & media.notna()
        media_activo  = bool(en_area.iloc[-1])
        media_cruce   = en_area & ~en_area.shift(1).fillna(False)
        media_frescura = _velas_desde_cruce(media_cruce) if media_activo else 999
        media_fresca  = media_activo and media_frescura <= UMBRAL_FRESCURA
    else:
        media_activo = media_fresca = False
        media_frescura = 999
    resultado["señales"]["S5_MEDIA"] = {
        "nombre": "Media K en área",
        "activa": media_activo,
        "fresca": media_fresca,
        "velas":  media_frescura if media_activo else None,
    }

    # ── S6: Bitman cambia a IMPULSO ALCISTA ───────────────────────────────
    if bitman_df is not None and not bitman_df.empty and "Bitman_Etiqueta" in bitman_df.columns:
        etiq           = bitman_df["Bitman_Etiqueta"]
        bitman_activo  = bool(etiq.iloc[-1] == "IMPULSO ALCISTA")
        bitman_cruce   = (etiq == "IMPULSO ALCISTA") & (etiq.shift(1) != "IMPULSO ALCISTA")
        bitman_frescura = _velas_desde_cruce(bitman_cruce) if bitman_activo else 999
        bitman_fresca  = bitman_activo and bitman_frescura <= UMBRAL_FRESCURA
    else:
        bitman_activo = bitman_fresca = False
        bitman_frescura = 999
    resultado["señales"]["S6_BITMAN"] = {
        "nombre": "Bitman impulso ↑",
        "activa": bitman_activo,
        "fresca": bitman_fresca,
        "velas":  bitman_frescura if bitman_activo else None,
    }

    # ── S7: BBWP pendiente positiva y > 20 ───────────────────────────────
    if len(bbwp_s.dropna()) >= 5:
        bbwp_last   = bbwp_s.dropna().iloc[-1]
        bbwp_prev   = bbwp_s.dropna().iloc[-4]
        s7_activa   = bool(bbwp_last > 20 and bbwp_last > bbwp_prev)
    else:
        s7_activa = False
    resultado["señales"]["S7_BBWP"] = {
        "nombre": "BBWP pendiente ↑",
        "activa": s7_activa,
        "fresca": False,   # informativa, sin frescura
        "velas":  None,
    }

    # ── S8: Volumen confirmatorio ─────────────────────────────────────────
    vol_ma   = volume.rolling(20).mean()
    vol_conf = (volume > vol_ma * 1.3) & (close > close.shift(1))
    # al menos 1 de las últimas 3 velas cumple
    s8_activa = bool(vol_conf.iloc[-3:].any())
    resultado["señales"]["S8_VOL"] = {
        "nombre": "Volumen confirm.",
        "activa": s8_activa,
        "fresca": False,
        "velas":  None,
    }

    # ── Conteo ────────────────────────────────────────────────────────────
    n_activas = sum(1 for s in resultado["señales"].values() if s["activa"])
    n_frescas = sum(
        1 for k, s in resultado["señales"].items()
        if s["fresca"] and k not in ("S7_BBWP", "S8_VOL")
    )
    resultado["n_activas"] = n_activas
    resultado["n_frescas"] = n_frescas

    # ── Señales informativas extra ────────────────────────────────────────
    informativas = []

    # SI1: precio en entorno MCG25
    mcg25_val = mcginley_dynamic(close, 25).iloc[-1]
    precio    = close.iloc[-1]
    if abs(precio / mcg25_val - 1) < 0.012:
        informativas.append("🟡 Precio en soporte MCG25")

    # SI2: precio en entorno EMA200
    ema200_val = EMAIndicator(close=close, window=200).ema_indicator().iloc[-1]
    if abs(precio / ema200_val - 1) < 0.015:
        informativas.append("🟡 Precio en soporte EMA200")

    # SI3: Atención Koncorde (azul+ y verde-)
    if not kdf.empty and "azul" in kdf.columns and "verde" in kdf.columns:
        if kdf["azul"].iloc[-1] > 0 and kdf["verde"].iloc[-1] < 0:
            informativas.append("⚠️ Atención Konkorde (azul+ verde-)")

    # SI4: Azul subiendo
    if not kdf.empty and "azul" in kdf.columns and len(kdf["azul"].dropna()) >= 4:
        if kdf["azul"].iloc[-1] > kdf["azul"].iloc[-4]:
            informativas.append("↑ Azul K acelerando")

    # SI5: Divergencia RSI alcista ≤15v
    rsi_div = detectar_divergencia_simple(df)
    hits    = rsi_div[rsi_div["divergencia_tipo"] == "alcista"]
    if not hits.empty:
        div_idx   = rsi_div.index.get_loc(hits.index[-1])
        div_velas = len(rsi_div) - 1 - div_idx
        if div_velas <= 15:
            informativas.append(f"🟢 Div alcista RSI ({div_velas}v)")

    resultado["informativas"] = informativas

    # ── Etiqueta final ────────────────────────────────────────────────────
    # Atención Konkorde es independiente
    if not kdf.empty and "azul" in kdf.columns and "verde" in kdf.columns:
        if kdf["azul"].iloc[-1] > 0 and kdf["verde"].iloc[-1] < 0:
            resultado["etiqueta"] = "⚠️ ATENCIÓN KONKORDE"
            # no return — seguimos calculando el resto para mostrar detalle

    if n_activas >= 6 and n_frescas >= 4:
        etiqueta = "🚀 POSITIVO CON MOMENTUM"
    elif n_activas >= 5 and n_frescas >= 2:
        etiqueta = "✅ POSITIVO"
    elif n_activas >= 5 and n_frescas < 2:
        etiqueta = "⏰ POSITIVO MADURO"
    elif n_activas >= 3 and n_frescas >= 2:
        etiqueta = "👀 EN DESARROLLO"
    elif n_activas >= 3:
        etiqueta = "👀 VIGILAR"
    else:
        etiqueta = "⛔ SIN SETUP"

    # Atención Konkorde tiene prioridad solo si no hay señal mejor
    if resultado["etiqueta"] == "⚠️ ATENCIÓN KONKORDE" and n_activas < 3:
        pass  # mantiene ATENCIÓN KONKORDE
    else:
        resultado["etiqueta"] = etiqueta

    # ── Texto razones ─────────────────────────────────────────────────────
    razones_parts = []
    for s in resultado["señales"].values():
        if s["activa"]:
            frescura_txt = f" 🔥{s['velas']}v" if s["fresca"] else ""
            razones_parts.append(f"✅ {s['nombre']}{frescura_txt}")
        else:
            razones_parts.append(f"❌ {s['nombre']}")

    if informativas:
        razones_parts.extend(informativas)

    resultado["razones"] = "  |  ".join(razones_parts)

    return resultado


# ══════════════════════════════════════════════════════════════════════════════
# SEMÁFORO DE SALIDA — para pestaña Mi Cartera
# ══════════════════════════════════════════════════════════════════════════════

def semaforo_salida(
    df:        pd.DataFrame,
    kdf:       pd.DataFrame,
    bitman_df: pd.DataFrame,
    macd_line: pd.Series,
    macd_sig:  pd.Series,
    pvi_s:     pd.Series,
    pvi_ema:   pd.Series,
) -> dict:
    """
    4 señales de salida — todas son CRUCES/CAMBIOS bajistas:
      SE1  AO cambia verde → rojo
      SE2  MACD cruza señal hacia abajo
      SE3  PVI cruza EMA25 hacia abajo
      SE4  Bitman cambia a RETROCESO o INDEFINICIÓN

    Etiquetas:
      0 señales  → 🟢 MANTENER
      1 señal    → 🟡 VIGILAR POSICIÓN
      2 señales  → 🟠 CONSIDERAR REDUCIR
      3+ señales → 🔴 SALIDA
    """
    resultado = {
        "señales": {},
        "n_salida": 0,
        "etiqueta": "🟢 MANTENER",
        "razones":  "",
    }

    high = df["High"]
    low  = df["Low"]
    n    = len(df)
    if n < 10:
        return resultado

    # ── SE1: AO cambia verde → rojo ───────────────────────────────────────
    ao_s      = awesome_osc(high, low)
    ao_verde  = ao_s > ao_s.shift(1)
    se1_activa = bool(not ao_verde.iloc[-1] and ao_verde.iloc[-2])
    resultado["señales"]["SE1_AO"] = {
        "nombre": "AO verde→rojo",
        "activa": se1_activa,
    }

    # ── SE2: MACD cruza señal hacia abajo ─────────────────────────────────
    diff        = macd_line - macd_sig
    se2_activa  = bool(diff.iloc[-1] < 0 and diff.iloc[-2] >= 0)
    resultado["señales"]["SE2_MACD"] = {
        "nombre": "MACD cruce ↓",
        "activa": se2_activa,
    }

    # ── SE3: PVI cruza EMA25 hacia abajo ──────────────────────────────────
    se3_activa = bool(
        pvi_s.iloc[-1] < pvi_ema.iloc[-1] and
        pvi_s.iloc[-2] >= pvi_ema.iloc[-2]
    )
    resultado["señales"]["SE3_PVI"] = {
        "nombre": "PVI cruza EMA25 ↓",
        "activa": se3_activa,
    }

    # ── SE4: Bitman cambia a RETROCESO o INDEFINICIÓN ─────────────────────
    if bitman_df is not None and not bitman_df.empty and len(bitman_df) >= 2:
        etiq_now  = bitman_df["Bitman_Etiqueta"].iloc[-1]
        etiq_prev = bitman_df["Bitman_Etiqueta"].iloc[-2]
        se4_activa = bool(
            etiq_now != "IMPULSO ALCISTA" and
            etiq_prev == "IMPULSO ALCISTA"
        )
    else:
        se4_activa = False
    resultado["señales"]["SE4_BITMAN"] = {
        "nombre": "Bitman pierde impulso",
        "activa": se4_activa,
    }

    # ── Conteo y etiqueta ─────────────────────────────────────────────────
    n_salida = sum(1 for s in resultado["señales"].values() if s["activa"])
    resultado["n_salida"] = n_salida

    if n_salida == 0:
        etiqueta = "🟢 MANTENER"
    elif n_salida == 1:
        etiqueta = "🟡 VIGILAR POSICIÓN"
    elif n_salida == 2:
        etiqueta = "🟠 CONSIDERAR REDUCIR"
    else:
        etiqueta = "🔴 SALIDA"
    resultado["etiqueta"] = etiqueta

    # ── Texto razones ─────────────────────────────────────────────────────
    partes = []
    for s in resultado["señales"].values():
        partes.append(
            f"🔴 {s['nombre']}" if s["activa"] else f"🟢 {s['nombre']} ok"
        )
    resultado["razones"] = "  |  ".join(partes)

    return resultado


# ══════════════════════════════════════════════════════════════════════════════
# GET SOVEREIGN DASHBOARD V2 — usa la nueva lógica de señales
# ══════════════════════════════════════════════════════════════════════════════

def get_sovereign_dashboard_v2(tickers: list, progress_cb=None) -> pd.DataFrame:
    """
    Versión 2 del dashboard con nuevo sistema de señales.
    Misma estructura de descarga e indicadores que v1.
    Solo cambia la lógica de scoring final.
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

            close  = df["Close"]
            volume = df["Volume"]
            high   = df["High"]
            low    = df["Low"]
            precio = close.iloc[-1]

            # indicadores (misma lógica que siempre)
            kdf = compute_blai5_koncorde(df, m=15)
            if kdf.empty:
                continue
            kdf = blai5_signals(kdf)

            bitman_df = clasificar_bitman(df)
            if bitman_df is None or bitman_df.empty:
                continue

            _, bbwp_s = calculate_bbwp(close, bb_len=13, lookback=252)

            macd_obj  = MACD(close=close)
            macd_line = macd_obj.macd()
            macd_sig  = macd_obj.macd_signal()

            pvi_s   = calculate_pvi(close, volume)
            pvi_ema = pvi_s.ewm(span=25, adjust=False).mean()

            rsi_s = RSIIndicator(close=close, window=14).rsi()

            mcg25_val = mcginley_dynamic(close, 25).iloc[-1]
            e200_val  = EMAIndicator(close=close, window=200).ema_indicator().iloc[-1]

            # nuevo scoring v2
            sv2 = calcular_señales_v2(
                df=df, kdf=kdf, bitman_df=bitman_df,
                bbwp_s=bbwp_s, macd_line=macd_line, macd_sig=macd_sig,
                pvi_s=pvi_s, pvi_ema=pvi_ema, rsi_s=rsi_s,
            )

            # tendencia (igual que antes)
            cerca_mcg  = mcg25_val * 0.988 <= precio <= mcg25_val * 1.012
            cerca_e200 = e200_val  * 0.985 <= precio <= e200_val  * 1.015
            s_mcg  = "🟡" if cerca_mcg  else ("🟢" if precio > mcg25_val else "🔴")
            s_e200 = "🟡" if cerca_e200 else ("🟢" if precio > e200_val  else "🔴")

            report.append({
                "Ticker":    t,
                "Precio":    f"{precio:.2f}",
                "Tendencia": f"MCG:{s_mcg} E200:{s_e200}",
                "Activas":   f"{sv2['n_activas']}/8",
                "Frescas":   f"{sv2['n_frescas']}/6",
                "Señal":     sv2["etiqueta"],
                "Detalle":   sv2["razones"],
            })

        except Exception as e:
            print(f"❌ {t}: {e}")
            continue

    result = pd.DataFrame(report)
    if not result.empty:
        orden = {
            "🚀 POSITIVO CON MOMENTUM": 0,
            "✅ POSITIVO":              1,
            "⚠️ ATENCIÓN KONKORDE":    2,
            "👀 EN DESARROLLO":         3,
            "⏰ POSITIVO MADURO":       4,
            "👀 VIGILAR":               5,
            "⛔ SIN SETUP":             6,
        }
        result["_sort"] = result["Señal"].map(orden).fillna(7)
        result = (result
                  .sort_values(["_sort", "Ticker"])
                  .drop(columns="_sort")
                  .reset_index(drop=True))
    return result
