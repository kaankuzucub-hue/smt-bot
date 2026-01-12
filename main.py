import sys
import traceback
import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.signal import argrelextrema
from datetime import datetime, time
import pytz
from itertools import combinations

# --- SETTINGS ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# --- MONEY MANAGEMENT ---
ACCOUNT_SIZE = 100000
RISK_AMOUNT = 1000
REWARD_RATIO = 2.0

# --- DATA RULES ---
STATS_DATA_PERIOD = "1y"
MAIN_INDEX = "TQQQ"

# --- SMT CONFIGURATION (HEPSİ DURUYOR) ---
SMT_CONFIG = {
    "SET_1": {"type": "standard", "name": "🔥 TQQQ TRIO", "ref": "TQQQ", "comps": ["SOXL", "NVDA"]},
    "SET_2": {"type": "standard", "name": "⚖️ TQQQ SEMI DUO", "ref": "TQQQ", "comps": ["SOXL"]},
    "SET_3": {"type": "cluster", "name": "⚔️ CHIP WARS (Matrix)", "peers": ["NVDA", "AVGO", "MU"]},
    "SET_4": {"type": "standard", "name": "🏥 SECTOR X-RAY", "ref": "TQQQ", "comps": ["XLK", "XLC", "XLY", "SMH"]}
}

# Timeframes
TF_MICRO = "5m"
TF_SCALP = "15m"
TF_SWING = "1h"

# --- THRESHOLDS ---
SCALP_PERCENTILE = 75
SWING_PERCENTILE = 90
FRESHNESS_LIMIT = 5

# --- HELPER FUNCTIONS ---
def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("!!! WARNING: Token or Chat ID missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"!!! Telegram Error: {e}")

def get_ny_time():
    ny_tz = pytz.timezone('America/New_York')
    return datetime.now(ny_tz)

def is_opening_range():
    now_ny = get_ny_time().time()
    return time(9, 30) <= now_ny <= time(11, 30)

def send_system_ok_message():
    now = get_ny_time()
    msg = (f"🟢 **SYSTEM OPERATIONAL** 🟢\n"
           f"🕒 NY Time: `{now.strftime('%H:%M')}`\n"
           f"✅ Bot: Active\n"
           f"🧠 Mode: MATH GOD (Markov/Hurst/FFT)")
    send_telegram(msg)

def safe_float(val):
    try:
        if isinstance(val, pd.Series): return float(val.iloc[0])
        return float(val)
    except: return 0.0

# ==========================================
# 🧠 LAYER 5: MATH GOD (YENİ EKLENDİ)
# ==========================================

# 1. HURST EXPONENT (Trend vs Mean Reversion)
def calculate_hurst(df, lags_count=20):
    try:
        ts = df['Close'].values
        lags = range(2, lags_count)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] * 2.0
        
        if hurst > 0.55: return f"🌊 **Trending ({hurst:.2f})**" # Trend Takip Et
        elif hurst < 0.45: return f"🪃 **Mean Rev ({hurst:.2f})**" # Tepeden Sat
        else: return f"🎲 **Random ({hurst:.2f})**" # İşlem Yapma
    except: return "N/A"

# 2. MARKOV CHAINS (Olasılık Hesabı)
def calculate_markov_prob(df):
    try:
        # Son 100 mumu al
        closes = df['Close'].pct_change().dropna().tail(100)
        # 1: Yükseliş, 0: Düşüş
        states = (closes > 0).astype(int)
        
        # Geçişleri say (0->1, 1->1 vs)
        trans_mat = np.zeros((2, 2))
        for i in range(len(states)-1):
            curr = states.iloc[i]
            next_s = states.iloc[i+1]
            trans_mat[curr][next_s] += 1
            
        # Olasılığa çevir
        # Şu anki durum ne? (Son mum yeşil mi kırmızı mı?)
        current_state = states.iloc[-1]
        
        row_sum = np.sum(trans_mat[current_state])
        if row_sum == 0: return "N/A"
        
        prob_bull = (trans_mat[current_state][1] / row_sum) * 100
        prob_bear = (trans_mat[current_state][0] / row_sum) * 100
        
        if prob_bull > 60: return f"🐂 **Bull Prob: %{prob_bull:.0f}**"
        elif prob_bear > 60: return f"🐻 **Bear Prob: %{prob_bear:.0f}**"
        else: return f"⚖️ **Neutral (%{prob_bull:.0f})**"
    except: return "N/A"

# 3. FFT (Fast Fourier Transform - Döngü Analizi)
def calculate_fft_cycle(df):
    try:
        # Trendi temizle (Detrend)
        closes = df['Close'].values
        linear_trend = np.linspace(closes[0], closes[-1], len(closes))
        detrended = closes - linear_trend
        
        # FFT Uygula
        fft_vals = np.fft.rfft(detrended)
        fft_freq = np.fft.rfftfreq(len(closes))
        
        # En güçlü frekansı bul (DC component 0 hariç)
        magnitudes = np.abs(fft_vals)
        peak_freq_idx = np.argmax(magnitudes[1:]) + 1
        peak_freq = fft_freq[peak_freq_idx]
        
        if peak_freq == 0: return "N/A"
        
        cycle_len = int(1 / peak_freq)
        
        # Döngünün neresindeyiz? (Basit yaklaşım)
        # Son X mumda fiyat dip mi yaptı tepe mi?
        return f"🔄 **Cycle: {cycle_len} Bars**"
    except: return "N/A"

# ==========================================
# 🏦 LAYER 4: INSTITUTIONAL FLOW & STATS
# ==========================================
def calculate_z_score(df, period=20):
    try:
        closes = df['Close']
        z_score = (closes.iloc[-1] - closes.rolling(period).mean().iloc[-1]) / closes.rolling(period).std().iloc[-1]
        if z_score > 3.0: return "🔥 **EXTREME (+3σ)**"
        elif z_score < -3.0: return "💎 **EXTREME (-3σ)**"
        elif z_score > 2.0: return "⚠️ **High (+2σ)**"
        elif z_score < -2.0: return "♻️ **Low (-2σ)**"
        else: return f"Neutral ({z_score:.1f}σ)"
    except: return "N/A"

def calculate_mfi(df, period=14):
    try:
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        mf = tp * df['Volume']
        pos = mf.where(tp > tp.shift(), 0).rolling(period).sum()
        neg = mf.where(tp < tp.shift(), 0).rolling(period).sum()
        return safe_float(100 - (100 / (1 + (pos/neg))).iloc[-1])
    except: return 50.0

def check_vwap_status(df):
    try:
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (tp * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
        dist = ((df['Close'].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]) * 100
        if dist > 2.0: return f"Expensive (+{dist:.1f}%)"
        elif dist < -2.0: return f"Cheap ({dist:.1f}%)"
        else: return "At VWAP"
    except: return "N/A"

# ==========================================
# 🧠 LAYER 3: QUANT RISK
# ==========================================
def calculate_atr(df, period=14):
    try:
        h_l = df['High'] - df['Low']
        h_c = (df['High'] - df['Close'].shift()).abs()
        l_c = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
        return safe_float(tr.rolling(period).mean().iloc[-1])
    except: return 0.0

def calculate_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = delta.where(delta>0, 0).rolling(period).mean()
        loss = (-delta.where(delta<0, 0)).rolling(period).mean()
        return 100 - (100 / (1 + gain/loss))
    except: return pd.Series([50]*len(series))

def get_vix_sentiment():
    try:
        vix = yf.download("^VIX", period="2d", interval="1d", progress=False).iloc[-1]['Close']
        vix = safe_float(vix)
        if vix > 25: return f"🌪️ **FEAR ({vix:.0f})**"
        return f"🌊 **Safe ({vix:.0f})**"
    except: return "N/A"

def find_nearest_fvg(df, direction):
    try:
        c, h, l = df['Close'].values, df['High'].values, df['Low'].values
        best_p, min_d = 0, 99999
        cp = c[-1]
        for i in range(len(c)-2, max(0, len(c)-20), -1):
            if direction == "SHORT" and l[i] > h[i+2]:
                gap = (l[i]+h[i+2])/2
                if gap < cp and (cp-gap) < min_d: min_d, best_p = cp-gap, gap
            elif direction == "LONG" and h[i] < l[i+2]:
                gap = (h[i]+l[i+2])/2
                if gap > cp and (gap-cp) < min_d: min_d, best_p = gap-cp, gap
        return f"${best_p:.2f}" if best_p else "None"
    except: return "Err"

# ==========================================
# 🔙 BACKTEST & PLAN
# ==========================================
def generate_trade_plan(price, direction, atr):
    if atr <= 0: return "N/A"
    sl = price + 1.5*atr if direction == "SHORT" else price - 1.5*atr
    tp = price - 3.0*atr if direction == "SHORT" else price + 3.0*atr
    return f"🛑 Stop: {sl:.2f}\n💰 Target: {tp:.2f} (1:2)"

def check_past_trade(df, entry_idx, direction, atr):
    try:
        ep = safe_float(df['Close'].iloc[entry_idx])
        sl = ep + 1.5*atr if direction == "SHORT" else ep - 1.5*atr
        tp = ep - 3.0*atr if direction == "SHORT" else ep + 3.0*atr
        
        future = df.iloc[entry_idx+1:]
        for i in range(len(future)):
            h, l = safe_float(future['High'].iloc[i]), safe_float(future['Low'].iloc[i])
            if direction == "SHORT":
                if l <= tp: return f"🏆 **WIN** (+${RISK_AMOUNT*REWARD_RATIO:,.0f})"
                if h >= sl: return f"❌ **LOSS** (-${RISK_AMOUNT:,.0f})"
            else:
                if h >= tp: return f"🏆 **WIN** (+${RISK_AMOUNT*REWARD_RATIO:,.0f})"
                if l <= sl: return f"❌ **LOSS** (-${RISK_AMOUNT:,.0f})"
        return "⏳ **PENDING**"
    except: return "N/A"

# --- CORE LOGIC ---
def get_data(symbol, interval):
    return yf.download(symbol, period=("1d" if interval=="5m" else "5d"), interval=interval, progress=False, auto_adjust=True)

def find_swings(df, order):
    if df is None or len(df)<10: return None, None, None, None
    c = df['Close'].values.flatten()
    mins = argrelextrema(c, np.less_equal, order=order)[0]
    maxs = argrelextrema(c, np.greater_equal, order=order)[0]
    if not len(mins) or not len(maxs): return None, None, None, None
    return df.iloc[mins]['Close'], df.iloc[maxs]['Close'], mins, maxs

def analyze_market_regime():
    print(f">>> Fetching {MAIN_INDEX}...")
    df = get_data(MAIN_INDEX, "1d")
    if df is None: return 0, "NO_DATA", 0
    # ... (Basit piyasa analizi kodu aynı)
    today = df.iloc[-1]
    return 0.0, "NORMAL", safe_float(today['Close']) # Basitleştirildi

def scan_smt_for_set(set_key, timeframe, market_status, market_change):
    config = SMT_CONFIG[set_key]
    is_cluster = config.get("type") == "cluster"
    header = "🌅 **OPENING RANGE**" if is_opening_range() else "⚡ **INTRADAY**"
    
    if is_cluster:
        peers = config["peers"]
        data = {}
        # VIX Tekrar çekmemek için basit tuttum
        for p in peers:
            df = get_data(p, timeframe)
            if df is None: continue
            l, h, l_idx, h_idx = find_swings(df, 2 if timeframe=="5m" else 3)
            if l is not None:
                atr = calculate_atr(df)
                data[p] = {"df":df, "l":l, "h":h, "l_idx":l_idx, "h_idx":h_idx, "atr":atr, "last":len(df)-1, "c":safe_float(df['Close'].iloc[-1])}
        
        if len(data) < 2: return
        for s1, s2 in combinations(data.keys(), 2):
            d1, d2 = data[s1], data[s2]
            
            # SHORT CHECK
            if (d1["last"]-d1["h_idx"][-1] <= FRESHNESS_LIMIT) and (d2["last"]-d2["h_idx"][-1] <= FRESHNESS_LIMIT):
                leader = s1 if d1["h"].iloc[-1] > d1["h"].iloc[-2] and d2["h"].iloc[-1] < d2["h"].iloc[-2] else \
                         s2 if d2["h"].iloc[-1] > d2["h"].iloc[-2] and d1["h"].iloc[-1] < d1["h"].iloc[-2] else None
                
                if leader:
                    laggard = s2 if leader == s1 else s1
                    main = data[leader]
                    
                    # --- MATH GOD CALCS ---
                    hurst = calculate_hurst(main["df"])
                    markov = calculate_markov_prob(main["df"])
                    cycle = calculate_fft_cycle(main["df"])
                    z = calculate_z_score(main["df"])
                    
                    # --- MSG ---
                    msg = (f"{header}\n⚔️ **CHIP WAR ({s1} vs {s2})**\n\n"
                           f"🚨 **ACTION: SHORT** 📉\n------------------------\n"
                           f"💪 **Strong:** {leader} (HH)\n🛑 **Weak:** {laggard} (LH)\n"
                           f"⏱️ **TF:** {timeframe}\n"
                           f"========================\n"
                           f"🧠 **MATH GOD MODE:**\n"
                           f"🎲 {markov}\n🌊 {hurst}\n{cycle}\n"
                           f"========================\n"
                           f"🏦 **INSTITUTIONAL:**\n"
                           f"📏 {z}\n"
                           f"{generate_trade_plan(main['c'], 'SHORT', main['atr'])}")
                    send_telegram(msg)
            
            # LONG CHECK (Benzer Mantık)
            if (d1["last"]-d1["l_idx"][-1] <= FRESHNESS_LIMIT) and (d2["last"]-d2["l_idx"][-1] <= FRESHNESS_LIMIT):
                leader = s1 if d1["l"].iloc[-1] < d1["l"].iloc[-2] and d2["l"].iloc[-1] > d2["l"].iloc[-2] else \
                         s2 if d2["l"].iloc[-1] < d2["l"].iloc[-2] and d1["l"].iloc[-1] > d1["l"].iloc[-2] else None
                
                if leader:
                    laggard = s2 if leader == s1 else s1
                    main = data[leader]
                    
                    hurst = calculate_hurst(main["df"])
                    markov = calculate_markov_prob(main["df"])
                    cycle = calculate_fft_cycle(main["df"])
                    z = calculate_z_score(main["df"])

                    msg = (f"{header}\n⚔️ **CHIP WAR ({s1} vs {s2})**\n\n"
                           f"🚨 **ACTION: LONG** 🚀\n------------------------\n"
                           f"📉 **Sweeping:** {leader} (LL)\n🛡️ **Holding:** {laggard} (HL)\n"
                           f"⏱️ **TF:** {timeframe}\n"
                           f"========================\n"
                           f"🧠 **MATH GOD MODE:**\n"
                           f"🎲 {markov}\n🌊 {hurst}\n{cycle}\n"
                           f"========================\n"
                           f"🏦 **INSTITUTIONAL:**\n"
                           f"📏 {z}\n"
                           f"{generate_trade_plan(main['c'], 'LONG', main['atr'])}")
                    send_telegram(msg)

    else: # STANDARD MODE
        ref = config["ref"]
        df = get_data(ref, timeframe)
        if df is None: return
        l, h, l_idx, h_idx = find_swings(df, 2 if timeframe=="5m" else 3)
        if l is None: return
        
        atr = calculate_atr(df)
        rsi = calculate_rsi(df['Close'])
        data_ref = {"l":l, "h":h, "l_idx":l_idx, "h_idx":h_idx, "c":safe_float(df['Close'].iloc[-1]), "last":len(df)-1}
        
        # MATH & QUANT
        hurst = calculate_hurst(df)
        markov = calculate_markov_prob(df)
        cycle = calculate_fft_cycle(df)
        z = calculate_z_score(df)
        mfi = calculate_mfi(df)
        
        comps = config["comps"]
        divs_short, divs_long = [], []
        
        for c in comps:
            df_c = get_data(c, timeframe)
            if df_c is None: continue
            lc, hc, _, _ = find_swings(df_c, 2 if timeframe=="5m" else 3)
            if lc is not None:
                if data_ref["h"].iloc[-1] > data_ref["h"].iloc[-2] and hc.iloc[-1] < hc.iloc[-2]: divs_short.append(c)
                if data_ref["l"].iloc[-1] < data_ref["l"].iloc[-2] and lc.iloc[-1] > lc.iloc[-2]: divs_long.append(c)

        # SEND MESSAGES
        # SHORT
        if divs_short and (data_ref["last"] - data_ref["h_idx"][-1] <= FRESHNESS_LIMIT):
            rsi_val = safe_float(rsi.iloc[data_ref["h_idx"][-1]])
            prev_rsi = safe_float(rsi.iloc[data_ref["h_idx"][-2]])
            is_div = rsi_val < prev_rsi
            icon = "💣" if is_div else "⚡"
            
            msg = (f"{header}\n{icon} **{config['name']} SHORT**\n\n"
                   f"🚨 **ACTION: SHORT** 📉\n------------------------\n"
                   f"📉 **Leader:** {ref} (HH)\n🛑 **Laggard:** {', '.join(divs_short)}\n"
                   f"🔋 **RSI:** {prev_rsi:.0f}->{rsi_val:.0f} ({'Div' if is_div else 'No Div'})\n"
                   f"========================\n"
                   f"🧠 **MATH GOD MODE:**\n"
                   f"🎲 {markov}\n🌊 {hurst}\n{cycle}\n"
                   f"========================\n"
                   f"🏦 **INSTITUTIONAL:**\n"
                   f"📏 {z}\n💸 **MFI:** {mfi:.0f}\n"
                   f"{generate_trade_plan(data_ref['c'], 'SHORT', atr)}")
            send_telegram(msg)

        # LONG
        if divs_long and (data_ref["last"] - data_ref["l_idx"][-1] <= FRESHNESS_LIMIT):
            rsi_val = safe_float(rsi.iloc[data_ref["l_idx"][-1]])
            prev_rsi = safe_float(rsi.iloc[data_ref["l_idx"][-2]])
            is_div = rsi_val > prev_rsi
            icon = "🚀" if is_div else "⚡"
            
            msg = (f"{header}\n{icon} **{config['name']} LONG**\n\n"
                   f"🚨 **ACTION: LONG** 🚀\n------------------------\n"
                   f"📈 **Leader:** {ref} (LL)\n🛡️ **Holding:** {', '.join(divs_long)}\n"
                   f"🔋 **RSI:** {prev_rsi:.0f}->{rsi_val:.0f} ({'Div' if is_div else 'No Div'})\n"
                   f"========================\n"
                   f"🧠 **MATH GOD MODE:**\n"
                   f"🎲 {markov}\n🌊 {hurst}\n{cycle}\n"
                   f"========================\n"
                   f"🏦 **INSTITUTIONAL:**\n"
                   f"📏 {z}\n💸 **MFI:** {mfi:.0f}\n"
                   f"{generate_trade_plan(data_ref['c'], 'LONG', atr)}")
            send_telegram(msg)

if __name__ == "__main__":
    try:
        print(">>> Loop Started...")
        send_system_ok_message()
        m_pct, m_stat, m_prc = analyze_market_regime()
        if m_stat != "NO_DATA":
            for s in SMT_CONFIG:
                try: scan_smt_for_set(s, TF_SCALP, m_stat, m_pct)
                except: pass
                try: scan_smt_for_set(s, TF_SWING, m_stat, m_pct)
                except: pass
            if is_opening_range():
                for s in SMT_CONFIG:
                    try: scan_smt_for_set(s, TF_MICRO, m_stat, m_pct)
                    except: pass
        print(">>> Loop Finished.")
    except Exception as e:
        print(f"CRITICAL: {e}")
        sys.exit(1)
