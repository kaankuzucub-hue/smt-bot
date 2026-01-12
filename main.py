import sys
import traceback
import os
import time as t_time
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.signal import argrelextrema
from datetime import datetime, time
import pytz
from itertools import combinations

# ==========================================
# ⚙️ CONFIGURATION & SETTINGS
# ==========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# --- RISK MANAGEMENT ---
ACCOUNT_SIZE = 100000
RISK_AMOUNT = 1000
REWARD_RATIO = 2.0

# --- DATA SETTINGS ---
STATS_DATA_PERIOD = "1y"
MAIN_INDEX = "TQQQ"

# --- STRATEGY CONFIGURATION ---
SMT_CONFIG = {
    "SET_1": {"type": "standard", "name": "🔥 TQQQ TRIO", "ref": "TQQQ", "comps": ["SOXL", "NVDA"]},
    "SET_2": {"type": "standard", "name": "⚖️ TQQQ SEMI DUO", "ref": "TQQQ", "comps": ["SOXL"]},
    "SET_3": {"type": "cluster", "name": "⚔️ CHIP WARS (Matrix)", "peers": ["NVDA", "AVGO", "MU"]},
    "SET_4": {"type": "standard", "name": "🏥 SECTOR X-RAY", "ref": "TQQQ", "comps": ["XLK", "XLC", "XLY", "SMH"]}
}

# --- TIMEFRAMES ---
TF_MICRO = "5m"
TF_SCALP = "15m"
TF_SWING = "1h"

# --- THRESHOLDS ---
SCALP_PERCENTILE = 75
SWING_PERCENTILE = 90
FRESHNESS_LIMIT = 5 # Bars

# ==========================================
# 🛠️ UTILITIES & ROBUSTNESS
# ==========================================
def retry_request(func, retries=3, delay=2):
    """Network işlemlerini garantiye alan retry mekanizması"""
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            print(f"⚠️ Network Warning (Attempt {i+1}/{retries}): {e}")
            t_time.sleep(delay)
    print("❌ Critical Network Failure.")
    return None

def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("!!! WARNING: Token or Chat ID missing.")
        return
    
    def _send():
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        resp = requests.post(url, data=data, timeout=10)
        resp.raise_for_status()
        return True

    retry_request(_send)

def get_ny_time():
    ny_tz = pytz.timezone('America/New_York')
    return datetime.now(ny_tz)

def is_opening_range():
    now_ny = get_ny_time().time()
    return time(9, 30) <= now_ny <= time(11, 30)

def safe_float(val):
    try:
        if isinstance(val, pd.Series):
            if val.empty: return 0.0
            return float(val.iloc[0])
        return float(val)
    except: return 0.0

def send_system_ok_message():
    now = get_ny_time()
    msg = (f"🟢 **SYSTEM OPERATIONAL** 🟢\n"
           f"🕒 NY Time: `{now.strftime('%H:%M')}`\n"
           f"✅ Bot: Active\n"
           f"🧠 Mode: UNLIMITED (Full Quant Stack)")
    send_telegram(msg)

# ==========================================
# 📈 QUANTITATIVE LIBRARY (INDICATORS)
# ==========================================
class QuantLib:
    @staticmethod
    def calculate_atr(df, period=14):
        try:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            return safe_float(atr)
        except: return 0.0

    @staticmethod
    def calculate_rsi(series, period=14):
        try:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except: return pd.Series([50]*len(series))

    @staticmethod
    def calculate_adx(df, period=14):
        """Trend Gücünü Ölçer. ADX > 25 ise Trend Var."""
        try:
            plus_dm = df['High'].diff()
            minus_dm = df['Low'].diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
            
            tr = QuantLib.calculate_atr(df, period=1) # True Range for calc
            
            atr = df['High'] # Placeholder, proper smoothing needed usually but simple rolling works for estimation
            # Simplified ADX for robustness in minimal pandas
            return 25.0 # Placeholder return to prevent crash if complex calc fails, implement logic if needed.
            # (Full ADX implementation is lengthy, sticking to simple trend check via EMA alignment usually better for bots)
        except: return 0.0

    @staticmethod
    def calculate_z_score(df, period=20):
        try:
            closes = df['Close']
            sma = closes.rolling(window=period).mean()
            std = closes.rolling(window=period).std()
            if std.iloc[-1] == 0: return "N/A"
            z = (closes.iloc[-1] - sma.iloc[-1]) / std.iloc[-1]
            
            if z > 3.0: return "🔥 **EXTREME O/B (+3σ)**"
            elif z > 2.0: return "⚠️ **Stretched (+2σ)**"
            elif z < -3.0: return "💎 **EXTREME O/S (-3σ)**"
            elif z < -2.0: return "♻️ **Undervalued (-2σ)**"
            return f"Neutral ({z:.2f}σ)"
        except: return "N/A"

    @staticmethod
    def calculate_mfi(df, period=14):
        try:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            mf = tp * df['Volume']
            pos_mf = mf.where(tp > tp.shift(), 0).rolling(period).sum()
            neg_mf = mf.where(tp < tp.shift(), 0).rolling(period).sum()
            return safe_float(100 - (100 / (1 + (pos_mf / neg_mf))).iloc[-1])
        except: return 50.0

    @staticmethod
    def calculate_rvol(df, period=20):
        """Relative Volume: Şu anki hacim ortalamanın kaç katı?"""
        try:
            vol_sma = df['Volume'].rolling(window=period).mean().iloc[-1]
            curr_vol = df['Volume'].iloc[-1]
            if vol_sma == 0: return 1.0
            rvol = curr_vol / vol_sma
            return safe_float(rvol)
        except: return 1.0

    @staticmethod
    def check_vwap(df):
        try:
            v = df['Volume'].values
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (tp * v).cumsum() / v.cumsum() # Intraday VWAP approximation
            # For rolling VWAP in this context (last 20 bars approximation):
            cum_v = df['Volume'].rolling(20).sum()
            cum_pv = (tp * df['Volume']).rolling(20).sum()
            rolling_vwap = cum_pv / cum_v
            
            curr = df['Close'].iloc[-1]
            vwap_val = rolling_vwap.iloc[-1]
            
            diff = ((curr - vwap_val) / vwap_val) * 100
            if diff > 2.0: return f"Expensive (+{diff:.1f}%)"
            elif diff < -2.0: return f"Cheap ({diff:.1f}%)"
            return "Fair Value"
        except: return "N/A"

    @staticmethod
    def detect_patterns(df):
        """Son mumda Price Action formasyonu var mı?"""
        try:
            o = df['Open'].iloc[-1]; c = df['Close'].iloc[-1]
            h = df['High'].iloc[-1]; l = df['Low'].iloc[-1]
            body = abs(c - o)
            total_range = h - l
            
            if total_range == 0: return "None"

            # PINBAR (Hammer/Shooting Star)
            upper_wick = h - max(c, o)
            lower_wick = min(c, o) - l
            
            if lower_wick > (body * 2) and upper_wick < body: return "🕯️ **Bullish Pinbar**"
            if upper_wick > (body * 2) and lower_wick < body: return "🕯️ **Bearish Pinbar**"
            
            # ENGULFING (Yutan Mum)
            prev_o = df['Open'].iloc[-2]; prev_c = df['Close'].iloc[-2]
            prev_body = abs(prev_c - prev_o)
            
            # Bearish Engulfing
            if c < o and prev_c > prev_o and c < prev_o and o > prev_c: return "🕯️ **Bearish Engulfing**"
            # Bullish Engulfing
            if c > o and prev_c < prev_o and c > prev_o and o < prev_c: return "🕯️ **Bullish Engulfing**"

            return "None"
        except: return "None"

# ==========================================
# 🔌 EXTERNAL DATA FETCHERS
# ==========================================
def get_vix_sentiment():
    def _fetch():
        vix = yf.download("^VIX", period="2d", interval="1d", progress=False, multi_level_index=False)
        if vix is None or len(vix) < 1: return "N/A"
        val = safe_float(vix['Close'].iloc[-1])
        if val > 25: return f"🌪️ **EXTREME ({val:.0f})**"
        elif val > 18: return f"⚠️ **High ({val:.0f})**"
        return f"🌊 **Safe ({val:.0f})**"
    return retry_request(_fetch) or "N/A"

def get_market_data(ticker, period, interval):
    def _fetch():
        return yf.download(ticker, period=period, interval=interval, progress=False, multi_level_index=False, auto_adjust=True)
    return retry_request(_fetch)

# ==========================================
# 🔙 BACKTEST & PLANNING ENGINE
# ==========================================
def check_past_trade(df, entry_idx, direction, atr):
    try:
        if entry_idx < 0 or entry_idx >= len(df): return "N/A"
        entry = safe_float(df['Close'].iloc[entry_idx])
        if atr <= 0: return "N/A"

        sl = entry + (atr*1.5) if direction == "SHORT" else entry - (atr*1.5)
        tp = entry - (atr*3.0) if direction == "SHORT" else entry + (atr*3.0)
        
        future = df.iloc[entry_idx+1:]
        if future.empty: return "⏳ **PENDING**"

        for i in range(len(future)):
            h = safe_float(future['High'].iloc[i])
            l = safe_float(future['Low'].iloc[i])
            
            if direction == "SHORT":
                if l <= tp: return f"🏆 **WIN** (+${RISK_AMOUNT*REWARD_RATIO:,.0f})"
                if h >= sl: return f"❌ **LOSS** (-${RISK_AMOUNT:,.0f})"
            else:
                if h >= tp: return f"🏆 **WIN** (+${RISK_AMOUNT*REWARD_RATIO:,.0f})"
                if l <= sl: return f"❌ **LOSS** (-${RISK_AMOUNT:,.0f})"
        return "⏳ **PENDING**"
    except: return "Err"

def find_fvg(df, direction):
    try:
        closes = df['Close'].values; highs = df['High'].values; lows = df['Low'].values
        best_gap = 0; min_dist = float('inf'); curr = closes[-1]
        
        # Scan last 20 candles
        for i in range(len(closes)-2, max(0, len(closes)-20), -1):
            gap_mid = 0
            if direction == "SHORT" and lows[i] > highs[i+2]: # Gap Down
                gap_mid = (lows[i] + highs[i+2]) / 2
                if gap_mid < curr: # Gap is below price (Target)
                     dist = curr - gap_mid
                     if dist < min_dist: min_dist, best_gap = dist, gap_mid
            
            elif direction == "LONG" and highs[i] < lows[i+2]: # Gap Up
                gap_mid = (highs[i] + lows[i+2]) / 2
                if gap_mid > curr: # Gap is above price (Target)
                    dist = gap_mid - curr
                    if dist < min_dist: min_dist, best_gap = dist, gap_mid
                    
        return f"${best_gap:.2f}" if best_gap != 0 else "None"
    except: return "Err"

# ==========================================
# 🕵️ CORE SCANNER LOGIC
# ==========================================
def analyze_market_regime():
    print(f">>> Fetching {MAIN_INDEX} Regime Data...")
    df = get_market_data(MAIN_INDEX, STATS_DATA_PERIOD, "1d")
    
    if df is None or len(df) < 10:
        print("!!! Regime Data Fail.")
        return 0, "NO_DATA", 0
    
    # Calculate Regime Percentiles
    df['Up'] = (df['High'] - df['Open']) / df['Open']
    df['Down'] = (df['Open'] - df['Low']) / df['Open']
    
    # Clean data
    up = df['Up'].dropna(); down = df['Down'].dropna()
    if up.empty: return 0, "NO_DATA", 0

    swing_up = np.percentile(up, SWING_PERCENTILE)
    swing_down = np.percentile(down, SWING_PERCENTILE)
    scalp_up = np.percentile(up, SCALP_PERCENTILE)
    scalp_down = np.percentile(down, SCALP_PERCENTILE)
    
    today = df.iloc[-1]
    o = safe_float(today['Open']); c = safe_float(today['Close'])
    pct = (c - o) / o if o != 0 else 0
    
    status = "NORMAL"
    if pct > 0:
        if pct > swing_up: status = "SWING_SHORT_ZONE"
        elif pct > scalp_up: status = "SCALP_SHORT_ZONE"
    elif pct < 0:
        d = abs(pct)
        if d > swing_down: status = "SWING_LONG_ZONE"
        elif d > scalp_down: status = "SCALP_LONG_ZONE"
        
    print(f">>> Regime: {status} ({pct:.2%})")
    return pct, status, c

def scan_strategy(set_key, timeframe, regime_status, regime_change):
    cfg = SMT_CONFIG[set_key]
    strat_type = cfg.get("type", "standard")
    
    # Adaptive Order: Less noise on higher TFs
    order = 2 if timeframe == "5m" else 3
    
    time_header = "🌅 **OPENING RANGE SNIPER**" if is_opening_range() else "⚡ **INTRADAY SCAN**"

    # --- CLUSTER LOGIC (Matrix) ---
    if strat_type == "cluster":
        peers = cfg["peers"]
        data = {}
        vix = get_vix_sentiment()
        
        # 1. Fetch & Analyze Peers
        for p in peers:
            df = get_market_data(p, "5d", timeframe)
            if df is None: continue
            
            close_v = df['Close'].values.flatten()
            mins = argrelextrema(close_v, np.less_equal, order=order)[0]
            maxs = argrelextrema(close_v, np.greater_equal, order=order)[0]
            
            if len(mins) < 2 or len(maxs) < 2: continue
            
            # Quant Calcs
            atr = QuantLib.calculate_atr(df)
            mfi = QuantLib.calculate_mfi(df)
            z_scr = QuantLib.calculate_z_score(df)
            vwap = QuantLib.check_vwap(df)
            rvol = QuantLib.calculate_rvol(df)
            pattern = QuantLib.detect_patterns(df)
            
            data[p] = {
                "H_new": safe_float(df.iloc[maxs[-1]]['Close']), "H_old": safe_float(df.iloc[maxs[-2]]['Close']),
                "L_new": safe_float(df.iloc[mins[-1]]['Close']), "L_old": safe_float(df.iloc[mins[-2]]['Close']),
                "H_idx": maxs[-1], "L_idx": mins[-1],
                "Close": safe_float(df['Close'].iloc[-1]),
                "ATR": atr, "MFI": mfi, "Z": z_scr, "VWAP": vwap, "RVOL": rvol, "PAT": pattern,
                "Last_Bar": len(df)-1, "DF": df
            }

        if len(data) < 2: return

        # 2. Compare Combinations
        for s1, s2 in combinations(sorted(data.keys()), 2):
            d1, d2 = data[s1], data[s2]
            
            # CHECK SHORT
            if (d1["Last_Bar"] - d1["H_idx"] <= FRESHNESS_LIMIT) and (d2["Last_Bar"] - d2["H_idx"] <= FRESHNESS_LIMIT):
                bearish = False
                leader, laggard = "", ""
                
                # SMT Logic: One makes HH, other makes LH
                if d1["H_new"] > d1["H_old"] and d2["H_new"] < d2["H_old"]:
                    bearish, leader, laggard = True, s1, s2
                elif d2["H_new"] > d2["H_old"] and d1["H_new"] < d1["H_old"]:
                    bearish, leader, laggard = True, s2, s1
                
                if bearish:
                    m = data[leader]
                    sl = m["Close"] + (m["ATR"]*1.5)
                    tp = m["Close"] - (m["ATR"]*REWARD_RATIO)
                    fvg = find_fvg(m["DF"], "SHORT")
                    prev_res = check_past_trade(m["DF"], m["H_idx"][-2], "SHORT", m["ATR"])
                    
                    msg = (f"{time_header}\n⚔️ **CHIP WAR ({s1} vs {s2})**\n\n"
                           f"🚨 **ACTION: SHORT** 📉\n------------------------\n"
                           f"💪 **Strong:** {leader} (HH)\n🛑 **Weak:** {laggard} (LH)\n"
                           f"🧠 **Pattern:** {m['PAT']}\n"
                           f"========================\n"
                           f"📊 **QUANT INSIGHTS:**\n🌪️ **VIX:** {vix}\n🧲 **FVG:** {fvg}\n"
                           f"🔊 **RVol:** {m['RVOL']:.1f}x\n"
                           f"🔙 **BACKTEST:** {prev_res}\n"
                           f"========================\n"
                           f"🏦 **FLOW:**\n📏 **Z:** {m['Z']}\n💸 **MFI:** {m['MFI']:.0f}\n⚖️ **VWAP:** {m['VWAP']}\n"
                           f"🛑 Stop: {sl:.2f}\n💰 Target: {tp:.2f}")
                    send_telegram(msg)

            # CHECK LONG
            if (d1["Last_Bar"] - d1["L_idx"] <= FRESHNESS_LIMIT) and (d2["Last_Bar"] - d2["L_idx"] <= FRESHNESS_LIMIT):
                bullish = False
                leader, laggard = "", ""
                
                # SMT Logic: One makes LL, other makes HL
                if d1["L_new"] < d1["L_old"] and d2["L_new"] > d2["L_old"]:
                    bullish, leader, laggard = True, s1, s2
                elif d2["L_new"] < d2["L_old"] and d1["L_new"] > d1["L_old"]:
                    bullish, leader, laggard = True, s2, s1
                
                if bullish:
                    m = data[leader]
                    sl = m["Close"] - (m["ATR"]*1.5)
                    tp = m["Close"] + (m["ATR"]*REWARD_RATIO)
                    fvg = find_fvg(m["DF"], "LONG")
                    prev_res = check_past_trade(m["DF"], m["L_idx"][-2], "LONG", m["ATR"])
                    
                    msg = (f"{time_header}\n⚔️ **CHIP WAR ({s1} vs {s2})**\n\n"
                           f"🚨 **ACTION: LONG** 🚀\n------------------------\n"
                           f"📉 **Sweep:** {leader} (LL)\n🛡️ **Hold:** {laggard} (HL)\n"
                           f"🧠 **Pattern:** {m['PAT']}\n"
                           f"========================\n"
                           f"📊 **QUANT INSIGHTS:**\n🌪️ **VIX:** {vix}\n🧲 **FVG:** {fvg}\n"
                           f"🔊 **RVol:** {m['RVOL']:.1f}x\n"
                           f"🔙 **BACKTEST:** {prev_res}\n"
                           f"========================\n"
                           f"🏦 **FLOW:**\n📏 **Z:** {m['Z']}\n💸 **MFI:** {m['MFI']:.0f}\n⚖️ **VWAP:** {m['VWAP']}\n"
                           f"🛑 Stop: {sl:.2f}\n💰 Target: {tp:.2f}")
                    send_telegram(msg)

    # --- STANDARD LOGIC (ETF/Index) ---
    else:
        ref = cfg["ref"]; comps = cfg["comps"]
        df_ref = get_market_data(ref, "5d", timeframe)
        if df_ref is None: return

        # Indicators
        rsi = QuantLib.calculate_rsi(df_ref['Close'])
        atr = QuantLib.calculate_atr(df_ref)
        z_scr = QuantLib.calculate_z_score(df_ref)
        mfi = QuantLib.calculate_mfi(df_ref)
        vwap = QuantLib.check_vwap(df_ref)
        rvol = QuantLib.calculate_rvol(df_ref)
        pat = QuantLib.detect_patterns(df_ref)
        vix = get_vix_sentiment()

        # Swings
        close_v = df_ref['Close'].values.flatten()
        mins = argrelextrema(close_v, np.less_equal, order=order)[0]
        maxs = argrelextrema(close_v, np.greater_equal, order=order)[0]
        
        if len(mins) < 2 or len(maxs) < 2: return
        
        # Reference Data Store
        ref_data = {
            "H_new": safe_float(df_ref.iloc[maxs[-1]]['Close']), "H_old": safe_float(df_ref.iloc[maxs[-2]]['Close']),
            "L_new": safe_float(df_ref.iloc[mins[-1]]['Close']), "L_old": safe_float(df_ref.iloc[mins[-2]]['Close']),
            "H_idx": maxs[-1], "L_idx": mins[-1],
            "Last_Bar": len(df_ref)-1,
            "Price": safe_float(df_ref['Close'].iloc[-1])
        }

        # RSI Div Check
        try:
            rsi_h_new = safe_float(rsi.iloc[maxs[-1]]); rsi_h_old = safe_float(rsi.iloc[maxs[-2]])
            rsi_l_new = safe_float(rsi.iloc[mins[-1]]); rsi_l_old = safe_float(rsi.iloc[mins[-2]])
        except: rsi_h_new=rsi_h_old=rsi_l_new=rsi_l_old=50

        # Scan Comps
        divergences = []
        # Check Short Divs
        if ref_data["H_new"] > ref_data["H_old"] and (ref_data["Last_Bar"] - ref_data["H_idx"] <= FRESHNESS_LIMIT):
            for c in comps:
                df_c = get_market_data(c, "5d", timeframe)
                if df_c is None: continue
                c_v = df_c['Close'].values.flatten()
                c_maxs = argrelextrema(c_v, np.greater_equal, order=order)[0]
                if len(c_maxs) >= 2:
                    h_n = safe_float(df_c.iloc[c_maxs[-1]]['Close'])
                    h_o = safe_float(df_c.iloc[c_maxs[-2]]['Close'])
                    if h_n < h_o: divergences.append(c) # Lower High = Divergence

            if divergences:
                rsi_div = (rsi_h_new < rsi_h_old)
                header = "💣 **RSI + SMT SETUP**" if rsi_div else "⚡ **STANDARD SMT**"
                rsi_msg = f"📉 **RSI Div:** {rsi_h_old:.0f}->{rsi_h_new:.0f}" if rsi_div else f"RSI: {rsi_h_new:.0f}"
                
                sl = ref_data["Price"] + (atr*1.5)
                tp = ref_data["Price"] - (atr*REWARD_RATIO)
                fvg = find_fvg(df_ref, "SHORT")
                prev_res = check_past_trade(df_ref, maxs[-2], "SHORT", atr)

                msg = (f"{time_header}\n{header}\n\n"
                       f"🚨 **ACTION: SHORT** 📉\n------------------------\n"
                       f"📉 **Leader:** {ref}\n🛑 **Weak:** {', '.join(divergences)}\n"
                       f"🔋 **Mom:** {rsi_msg}\n🧠 **Pattern:** {pat}\n"
                       f"========================\n"
                       f"📊 **QUANT LAYER:**\n🌪️ **VIX:** {vix}\n🧲 **FVG:** {fvg}\n"
                       f"🔊 **RVol:** {rvol:.1f}x\n"
                       f"🔙 **BACKTEST:** {prev_res}\n"
                       f"========================\n"
                       f"🏦 **FLOW:**\n📏 **Z:** {z_scr}\n💸 **MFI:** {mfi:.0f}\n⚖️ **VWAP:** {vwap}\n"
                       f"🛑 Stop: {sl:.2f}\n💰 Target: {tp:.2f}")
                send_telegram(msg)

        # Check Long Divs
        if ref_data["L_new"] < ref_data["L_old"] and (ref_data["Last_Bar"] - ref_data["L_idx"] <= FRESHNESS_LIMIT):
            for c in comps:
                df_c = get_market_data(c, "5d", timeframe)
                if df_c is None: continue
                c_v = df_c['Close'].values.flatten()
                c_mins = argrelextrema(c_v, np.less_equal, order=order)[0]
                if len(c_mins) >= 2:
                    l_n = safe_float(df_c.iloc[c_mins[-1]]['Close'])
                    l_o = safe_float(df_c.iloc[c_mins[-2]]['Close'])
                    if l_n > l_o: divergences.append(c) # Higher Low = Divergence

            if divergences:
                rsi_div = (rsi_l_new > rsi_l_old)
                header = "🚀 **RSI + SMT SETUP**" if rsi_div else "⚡ **STANDARD SMT**"
                rsi_msg = f"📈 **RSI Div:** {rsi_l_old:.0f}->{rsi_l_new:.0f}" if rsi_div else f"RSI: {rsi_l_new:.0f}"
                
                sl = ref_data["Price"] - (atr*1.5)
                tp = ref_data["Price"] + (atr*REWARD_RATIO)
                fvg = find_fvg(df_ref, "LONG")
                prev_res = check_past_trade(df_ref, mins[-2], "LONG", atr)

                msg = (f"{time_header}\n{header}\n\n"
                       f"🚨 **ACTION: LONG** 🚀\n------------------------\n"
                       f"📈 **Leader:** {ref}\n💪 **Hold:** {', '.join(divergences)}\n"
                       f"🔋 **Mom:** {rsi_msg}\n🧠 **Pattern:** {pat}\n"
                       f"========================\n"
                       f"📊 **QUANT LAYER:**\n🌪️ **VIX:** {vix}\n🧲 **FVG:** {fvg}\n"
                       f"🔊 **RVol:** {rvol:.1f}x\n"
                       f"🔙 **BACKTEST:** {prev_res}\n"
                       f"========================\n"
                       f"🏦 **FLOW:**\n📏 **Z:** {z_scr}\n💸 **MFI:** {mfi:.0f}\n⚖️ **VWAP:** {vwap}\n"
                       f"🛑 Stop: {sl:.2f}\n💰 Target: {tp:.2f}")
                send_telegram(msg)

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
if __name__ == "__main__":
    print(">>> Initializing Hedge Fund Grade Bot...")
    try:
        send_system_ok_message()
        m_change, m_status, m_price = analyze_market_regime()
        
        if m_status != "NO_DATA":
            sets = ["SET_1", "SET_2", "SET_3", "SET_4"]
            
            # 1. Opening Range Priority
            if is_opening_range():
                print(">>> 🌅 Scanning Opening Range (5m)...")
                for s in sets:
                    try: scan_strategy(s, TF_MICRO, m_status, m_change)
                    except Exception as e: print(f"Error {s} 5m: {e}")
            
            # 2. Intraday Scans
            print(">>> ⚡ Scanning Intraday (15m & 1h)...")
            for s in sets:
                try: scan_strategy(s, TF_SCALP, m_status, m_change)
                except Exception as e: print(f"Error {s} 15m: {e}")
                
                try: scan_strategy(s, TF_SWING, m_status, m_change)
                except Exception as e: print(f"Error {s} 1h: {e}")

        print(">>> Cycle Complete.")
        
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        traceback.print_exc()
        sys.exit(1)
