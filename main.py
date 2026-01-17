import sys
import traceback
import os
import time
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, time as dtime
import pytz
from itertools import combinations

# --- GLOBAL STATE (MENÜ İÇİN) ---
CURRENT_FILTER = "ALL"
LAST_UPDATE_ID = 0

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

# --- SMT CONFIGURATION (AYNEN KORUNDU) ---
SMT_CONFIG = {
    "SET_1": {"type": "standard", "name": "🔥 TQQQ TRIO", "ref": "TQQQ", "comps": ["SOXL", "NVDA"]},
    "SET_2": {"type": "standard", "name": "⚖️ TQQQ SEMI DUO", "ref": "TQQQ", "comps": ["SOXL"]},
    "SET_3": {"type": "cluster", "name": "⚔️ CHIP WARS (Matrix)", "peers": ["NVDA", "AVGO", "MU"]},
    "SET_4": {"type": "standard", "name": "🏥 SECTOR X-RAY", "ref": "TQQQ", "comps": ["XLK", "XLC", "XLY", "SMH"]},
    "SET_X9191": {
        "type": "cluster",
        "name": "👽 PROTOCOL X-9191",
        "peers": ["TQQQ", "XLK", "SMH"] 
    }
}

# Timeframes
TF_MICRO = "5m"
TF_SCALP = "15m"
TF_SWING = "1h"

# --- THRESHOLDS ---
SCALP_PERCENTILE = 75
SWING_PERCENTILE = 90
FRESHNESS_LIMIT = 5

# --- TELEGRAM MENU & LISTENER FUNCTIONS ---

def send_control_panel():
    """Butonlu paneli ve menüyü gönderir"""
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    keyboard = {
        "inline_keyboard": [
            [
                {"text": "⚡ 5m", "callback_data": "5m"},
                {"text": "🎯 15m", "callback_data": "15m"},
                {"text": "⏳ 1h", "callback_data": "1h"}
            ],
            [{"text": "👁️ SHOW ALL (Reset)", "callback_data": "ALL"}],
            [{"text": "📊 Status", "callback_data": "STATUS"}]
        ]
    }
    msg_text = (f"🎛️ **TQQQ QUANT PANEL**\n"
                f"Current Filter: **{CURRENT_FILTER}**\n"
                f"Type `/menu` to bring this back.")
    data = {"chat_id": CHAT_ID, "text": msg_text, "parse_mode": "Markdown", "reply_markup": json.dumps(keyboard)}
    try: requests.post(url, data=data, timeout=10)
    except: pass

def check_updates():
    """Buton tıklamalarını ve /menu komutunu dinler"""
    global LAST_UPDATE_ID, CURRENT_FILTER
    if not TELEGRAM_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    try:
        resp = requests.get(url, params={"offset": LAST_UPDATE_ID + 1, "timeout": 1}, timeout=2).json()
        if "result" in resp:
            for update in resp["result"]:
                LAST_UPDATE_ID = update["update_id"]
                # Komut Kontrolü
                if "message" in update and "text" in update["message"]:
                    if update["message"]["text"] == "/menu":
                        send_control_panel()
                # Buton Kontrolü
                elif "callback_query" in update:
                    selection = update["callback_query"]["data"]
                    if selection in ["5m", "15m", "1h", "ALL"]:
                        CURRENT_FILTER = selection
                        send_telegram(f"⚙️ **FILTER:** Showing **{CURRENT_FILTER}**")
                    elif selection == "STATUS":
                        send_telegram(f"✅ Bot Online\nFilter: {CURRENT_FILTER}\nModule: X-9191")
                    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery", 
                                  data={"callback_query_id": update["callback_query"]["id"]})
    except: pass

# --- ORIGINAL HELPER FUNCTIONS ---
def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, data=data, timeout=10)
    except: pass

def get_ny_time():
    ny_tz = pytz.timezone('America/New_York')
    return datetime.now(ny_tz)

def is_opening_range():
    now_ny = get_ny_time().time()
    return dtime(9, 30) <= now_ny <= dtime(11, 30)

def send_system_ok_message():
    now = get_ny_time()
    msg = (f"🟢 **SYSTEM OPERATIONAL** 🟢\n"
           f"🕒 NY Time: `{now.strftime('%H:%M')}`\n"
           f"✅ Bot: Active\n"
           f"👽 Module: X-9191 LOADED")
    send_telegram(msg)
    send_control_panel()

def safe_float(val):
    try:
        if isinstance(val, pd.Series): 
            if val.empty: return 0.0
            return float(val.iloc[0])
        return float(val)
    except: return 0.0

# ==========================================
# 🧠 MATH GOD & QUANT LAYERS (AYNEN KORUNDU)
# ==========================================
def check_trend_bias(df):
    try:
        if len(df) < 200: return "NEUTRAL"
        ema200 = df['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
        current_price = safe_float(df['Close'].iloc[-1])
        return "BULLISH" if current_price > ema200 else "BEARISH"
    except: return "NEUTRAL"

def calculate_hurst(df, lags_count=20):
    try:
        ts = df['Close'].values.flatten()
        if len(ts) < lags_count + 5: return "N/A"
        lags = range(2, lags_count)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] * 2.0
        if hurst > 0.55: return f"🌊 **Trending ({hurst:.2f})**"
        elif hurst < 0.45: return f"🪃 **Mean Rev ({hurst:.2f})**"
        else: return f"🎲 **Random ({hurst:.2f})**"
    except: return "N/A"

def calculate_markov_prob(df):
    try:
        closes = df['Close'].pct_change().dropna().tail(100)
        if len(closes) < 10: return "N/A"
        states = (closes > 0).astype(int)
        trans_mat = np.zeros((2, 2))
        for i in range(len(states)-1): trans_mat[states.iloc[i]][states.iloc[i+1]] += 1
        current_state = states.iloc[-1]
        row_sum = np.sum(trans_mat[current_state])
        if row_sum == 0: return "N/A"
        prob_bull = (trans_mat[current_state][1] / row_sum) * 100
        prob_bear = (trans_mat[current_state][0] / row_sum) * 100
        if prob_bull > 60: return f"🐂 **Bull Prob: %{prob_bull:.0f}**"
        elif prob_bear > 60: return f"🐻 **Bear Prob: %{prob_bear:.0f}**"
        else: return f"⚖️ **Neutral (%{prob_bull:.0f})**"
    except: return "N/A"

def calculate_fft_cycle(df):
    try:
        closes = df['Close'].values.flatten()
        if len(closes) < 30: return "N/A"
        detrended = closes - np.linspace(closes[0], closes[-1], len(closes))
        fft_vals = np.fft.rfft(detrended)
        magnitudes = np.abs(fft_vals)
        peak_freq_idx = np.argmax(magnitudes[1:]) + 1
        return f"🔄 **Cycle: ~{int(len(closes) / peak_freq_idx)} Bars**"
    except: return "N/A"

def calculate_z_score(df, period=20):
    try:
        closes = df['Close']
        if len(closes) < period: return "N/A"
        z = (closes.iloc[-1] - closes.rolling(period).mean().iloc[-1]) / closes.rolling(period).std().iloc[-1]
        if z > 3.0: return "🔥 **EXTREME (+3σ)**"
        elif z < -3.0: return "💎 **EXTREME (-3σ)**"
        elif z > 2.0: return "⚠️ **High (+2σ)**"
        elif z < -2.0: return "♻️ **Low (-2σ)**"
        else: return f"Neutral ({z:.1f}σ)"
    except: return "N/A"

def calculate_mfi(df, period=14):
    try:
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        mf = tp * df['Volume']
        pos = mf.where(tp.diff() > 0, 0).rolling(period).sum()
        neg = mf.where(tp.diff() < 0, 0).rolling(period).sum()
        if neg.iloc[-1] == 0: return 50.0
        return safe_float(100 - (100 / (1 + (pos.iloc[-1] / neg.iloc[-1]))))
    except: return 50.0

def check_vwap_status(df):
    try:
        v = df['Volume']
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (tp * v).rolling(20).sum() / v.rolling(20).sum()
        dist = ((df['Close'].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]) * 100
        if dist > 2.0: return f"Expensive (+{dist:.1f}%)"
        elif dist < -2.0: return f"Cheap ({dist:.1f}%)"
        else: return "At VWAP"
    except: return "N/A"

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
        if loss.iloc[-1] == 0: return 50.0
        return 100 - (100 / (1 + gain/loss))
    except: return pd.Series([50]*len(series))

def get_vix_sentiment():
    try:
        vix = yf.download("^VIX", period="2d", interval="1d", progress=False)
        val = safe_float(vix['Close'].iloc[-1])
        return f"🌪️ **FEAR ({val:.0f})**" if val > 25 else f"🌊 **Safe ({val:.0f})**"
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

def generate_trade_plan(price, direction, atr):
    if atr <= 0: return "N/A"
    sl = price + 1.5*atr if direction == "SHORT" else price - 1.5*atr
    tp = price - 3.0*atr if direction == "SHORT" else price + 3.0*atr
    return f"🛑 Stop: {sl:.2f}\n💰 Target: {tp:.2f} (1:2)"

def check_past_trade(df, entry_idx, direction, atr):
    try:
        if atr <= 0: return "N/A"
        ep = safe_float(df['Close'].iloc[entry_idx])
        sl = ep + 1.5*atr if direction == "SHORT" else ep - 1.5*atr
        tp = ep - 3.0*atr if direction == "SHORT" else ep + 3.0*atr
        future = df.iloc[entry_idx+1:]
        if len(future) == 0: return "⏳ **JUST OPENED**"
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
    try:
        df = yf.download(symbol, period="1y", interval=interval, progress=False, auto_adjust=True)
        if df is None or len(df) < 2: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

def find_swings(df, order):
    if df is None or len(df)<10: return None, None, None, None
    try:
        c = df['Close'].values.flatten()
        mins = argrelextrema(c, np.less_equal, order=order)[0]
        maxs = argrelextrema(c, np.greater_equal, order=order)[0]
        return df.iloc[mins]['Close'], df.iloc[maxs]['Close'], mins, maxs
    except: return None, None, None, None

def scan_smt_for_set(set_key, timeframe, market_status, market_change):
    config = SMT_CONFIG[set_key]
    is_cluster = config.get("type") == "cluster"
    header = "🌅 **OPENING RANGE**" if is_opening_range() else "⚡ **INTRADAY**"
    
    # --- MENÜ FİLTRELEME KONTROLÜ ---
    if CURRENT_FILTER != "ALL" and CURRENT_FILTER != timeframe:
        return # Filtreye takıldıysa hesaplama yapma, çık.

    if is_cluster:
        peers = config["peers"]
        data = {}
        vix_msg = get_vix_sentiment()
        for p in peers:
            df = get_data(p, timeframe)
            if df is None: continue
            trend_bias = check_trend_bias(df)
            l, h, l_idx, h_idx = find_swings(df, 2 if timeframe=="5m" else 3)
            if l is not None:
                atr = calculate_atr(df)
                data[p] = {"df":df, "l":l, "h":h, "l_idx":l_idx, "h_idx":h_idx, "atr":atr, "last":len(df)-1, "c":safe_float(df['Close'].iloc[-1]), "bias": trend_bias}
        
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
                    action_txt = "🚨 **ACTION: SHORT (⚠️ RISKY)** 📉" if main["bias"] == "BULLISH" else "🚨 **ACTION: SHORT** 📉"
                    hurst, markov, cycle, z, mfi, vwap_st = calculate_hurst(main["df"]), calculate_markov_prob(main["df"]), calculate_fft_cycle(main["df"]), calculate_z_score(main["df"]), calculate_mfi(main["df"]), check_vwap_status(main["df"])
                    msg = (f"{header}\n{config['name']} ({s1} vs {s2})\n\n{action_txt}\n🛡️ **Status:** {main['bias']}\n💪 **Strong:** {leader}\n⏱️ **TF:** {timeframe}\n🧠 **MATH GOD:** {markov} | {hurst}\n📊 **QUANT:** VIX: {vix_msg} | FVG: {find_nearest_fvg(main['df'], 'SHORT')}\n{generate_trade_plan(main['c'], 'SHORT', main['atr'])}")
                    send_telegram(msg)
            
            # LONG CHECK
            if (d1["last"]-d1["l_idx"][-1] <= FRESHNESS_LIMIT) and (d2["last"]-d2["l_idx"][-1] <= FRESHNESS_LIMIT):
                leader = s1 if d1["l"].iloc[-1] < d1["l"].iloc[-2] and d2["l"].iloc[-1] > d2["l"].iloc[-2] else \
                         s2 if d2["l"].iloc[-1] < d2["l"].iloc[-2] and d1["l"].iloc[-1] > d1["l"].iloc[-2] else None
                if leader:
                    laggard = s2 if leader == s1 else s1
                    main = data[leader]
                    action_txt = "🚨 **ACTION: LONG (⚠️ RISKY)** 🚀" if main["bias"] == "BEARISH" else "🚨 **ACTION: LONG** 🚀"
                    hurst, markov, cycle, z, mfi, vwap_st = calculate_hurst(main["df"]), calculate_markov_prob(main["df"]), calculate_fft_cycle(main["df"]), calculate_z_score(main["df"]), calculate_mfi(main["df"]), check_vwap_status(main["df"])
                    msg = (f"{header}\n{config['name']} ({s1} vs {s2})\n\n{action_txt}\n🛡️ **Status:** {main['bias']}\n📈 **Sweeping:** {leader}\n⏱️ **TF:** {timeframe}\n🧠 **MATH GOD:** {markov} | {hurst}\n📊 **QUANT:** VIX: {vix_msg} | FVG: {find_nearest_fvg(main['df'], 'LONG')}\n{generate_trade_plan(main['c'], 'LONG', main['atr'])}")
                    send_telegram(msg)

    else: # STANDARD MODE
        ref = config["ref"]
        df = get_data(ref, timeframe)
        if df is None: return
        l, h, l_idx, h_idx = find_swings(df, 2 if timeframe=="5m" else 3)
        if l is None: return
        data_ref = {"l":l, "h":h, "l_idx":l_idx, "h_idx":h_idx, "c":safe_float(df['Close'].iloc[-1]), "last":len(df)-1}
        comps = config["comps"]
        divs_short, divs_long = [], []
        for c in comps:
            df_c = get_data(c, timeframe)
            if df_c is None: continue
            lc, hc, _, _ = find_swings(df_c, 2 if timeframe=="5m" else 3)
            if lc is not None:
                if data_ref["h"].iloc[-1] > data_ref["h"].iloc[-2] and hc.iloc[-1] < hc.iloc[-2]: divs_short.append(c)
                if data_ref["l"].iloc[-1] < data_ref["l"].iloc[-2] and lc.iloc[-1] > lc.iloc[-2]: divs_long.append(c)

        if divs_short and (data_ref["last"] - data_ref["h_idx"][-1] <= FRESHNESS_LIMIT):
            send_telegram(f"{header}\n⚡ **{config['name']} SHORT**\nLeader: {ref}\nLaggards: {', '.join(divs_short)}\nTF: {timeframe}")
        if divs_long and (data_ref["last"] - data_ref["l_idx"][-1] <= FRESHNESS_LIMIT):
            send_telegram(f"{header}\n🚀 **{config['name']} LONG**\nLeader: {ref}\nHolding: {', '.join(divs_long)}\nTF: {timeframe}")

# --- MAIN LOOP ---
if __name__ == "__main__":
    try:
        print(">>> Bot Started...")
        send_system_ok_message()
        
        while True:
            # Piyasayı tara
            strats = ["SET_1", "SET_2", "SET_3", "SET_4", "SET_X9191"]
            if is_opening_range():
                for s in strats: 
                    scan_smt_for_set(s, TF_MICRO, "NORMAL", 0)
                    check_updates() # Her taramada buton kontrolü
            
            for s in strats:
                for tf in [TF_SCALP, TF_SWING]:
                    scan_smt_for_set(s, tf, "NORMAL", 0)
                    check_updates()
            
            # Dinlenme Modu (AMA BUTONLARI DİNLEYEREK)
            for _ in range(60): # 1 Dakika bekleme
                check_updates()
                time.sleep(1)
                
    except Exception as e:
        print(f"CRITICAL: {e}")
        traceback.print_exc()
        sys.exit(1)
