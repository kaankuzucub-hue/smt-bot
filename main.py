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

# --- SETTINGS ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# --- GLOBAL STATE (MENU SEÇİMİ İÇİN) ---
# Varsayılan: "ALL" (Hepsini göster)
# Seçenekler: "5m", "15m", "1h", "ALL"
CURRENT_FILTER = "ALL"
LAST_UPDATE_ID = 0

# --- MONEY MANAGEMENT ---
ACCOUNT_SIZE = 100000
RISK_AMOUNT = 1000
REWARD_RATIO = 2.0

# --- DATA RULES ---
STATS_DATA_PERIOD = "1y" 
MAIN_INDEX = "TQQQ"

# --- SMT CONFIGURATION (TQQQ EDITION) ---
SMT_CONFIG = {
    "SET_1": {
        "type": "standard", 
        "name": "👯 TQ/SQ TWINS", 
        "ref": "TQQQ", 
        "comps": ["SQQQ"] 
    },
    "SET_2": {
        "type": "standard", 
        "name": "🤖 AI LEADERS", 
        "ref": "TQQQ", 
        "comps": ["NVDA", "SOXL"]
    },
    "SET_3": {
        "type": "cluster", 
        "name": "⚔️ TECH TRINITY", 
        "peers": ["TQQQ", "SOXL", "FNGU"]
    }
}

# Timeframes
TF_MICRO = "5m"
TF_SCALP = "15m"
TF_SWING = "1h"

FRESHNESS_LIMIT = 5

# --- TELEGRAM MENU & UPDATE HANDLER ---

def send_control_panel():
    """Kullanıcıya butonlu menüyü gönderir"""
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    
    # Buton Tasarımı
    keyboard = {
        "inline_keyboard": [
            [
                {"text": "⚡ 5m Only", "callback_data": "5m"},
                {"text": "🎯 15m Only", "callback_data": "15m"},
                {"text": "⏳ 1h Only", "callback_data": "1h"}
            ],
            [
                {"text": "👁️ SHOW ALL (Reset)", "callback_data": "ALL"}
            ],
            [
                {"text": "📊 Status Check", "callback_data": "STATUS"}
            ]
        ]
    }
    
    msg_text = (f"🎛️ **CONTROL PANEL**\n"
                f"Currently Filtering: **{CURRENT_FILTER}**\n"
                f"Bot runs all calcs in background. Select what you want to SEE:")
    
    data = {
        "chat_id": CHAT_ID, 
        "text": msg_text, 
        "parse_mode": "Markdown",
        "reply_markup": json.dumps(keyboard)
    }
    try: requests.post(url, data=data, timeout=10)
    except: pass

def check_updates():
    """Telegram'dan gelen buton tıklamalarını dinler"""
    global LAST_UPDATE_ID, CURRENT_FILTER
    
    if not TELEGRAM_TOKEN: return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    params = {"offset": LAST_UPDATE_ID + 1, "timeout": 1} # Hizli kontrol
    
    try:
        resp = requests.get(url, params=params, timeout=2)
        data = resp.json()
        
        if "result" in data:
            for update in data["result"]:
                LAST_UPDATE_ID = update["update_id"]
                
                # Eger bir butona basildiysa (Callback Query)
                if "callback_query" in update:
                    cb_id = update["callback_query"]["id"]
                    selection = update["callback_query"]["data"]
                    
                    if selection in ["5m", "15m", "1h", "ALL"]:
                        CURRENT_FILTER = selection
                        ack_text = f"✅ Filter Switched to: {selection}"
                    elif selection == "STATUS":
                        ack_text = f"Bot Running... Filter: {CURRENT_FILTER}"
                    else:
                        ack_text = "Unknown Command"
                        
                    # Telegrama "Islem Tamam" sinyali gonder (loading donmesin)
                    requests.post(
                        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
                        data={"callback_query_id": cb_id, "text": ack_text}
                    )
                    
                    # Kullaniciya bilgi mesaji
                    send_telegram(f"⚙️ **SYSTEM UPDATED:** Showing **{CURRENT_FILTER}** signals only.")
                    
    except Exception as e:
        pass # Hata olursa donguyu kirma, sessizce devam et

def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, data=data, timeout=10)
    except: pass

# --- STANDARD FUNCTIONS ---
def get_ny_time():
    ny_tz = pytz.timezone('America/New_York')
    return datetime.now(ny_tz)

def is_opening_range():
    now_ny = get_ny_time().time()
    return dtime(9, 30) <= now_ny <= dtime(11, 30)

def safe_float(val):
    try:
        if isinstance(val, pd.Series): return float(val.iloc[0]) if not val.empty else 0.0
        return float(val)
    except: return 0.0

# --- MATH GOD & QUANT LAYERS ---
# (Önceki kodların aynısı - Kısaltılmıştır, mantık değişmedi)
def check_trend_bias(df):
    try:
        if len(df) < 200: return "NEUTRAL"
        ema200 = df['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
        return "BULLISH" if safe_float(df['Close'].iloc[-1]) > ema200 else "BEARISH"
    except: return "NEUTRAL"

def calculate_hurst(df):
    # Hurst Implementation
    try:
        ts = df['Close'].values.flatten()
        if len(ts) < 25: return "N/A"
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] * 2.0
        if hurst > 0.55: return f"🌊 Trend ({hurst:.2f})"
        elif hurst < 0.45: return f"🪃 MeanRev ({hurst:.2f})"
        else: return f"🎲 Rand ({hurst:.2f})"
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
        if prob_bull > 60: return f"🐂 Bull %{prob_bull:.0f}"
        elif prob_bear > 60: return f"🐻 Bear %{prob_bear:.0f}"
        else: return "⚖️ Neut"
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

# --- CORE LOGIC ---
def get_data(symbol, interval):
    try:
        df = yf.download(symbol, period=("1d" if interval=="5m" else "5d"), interval=interval, progress=False, auto_adjust=True)
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
        if not len(mins) or not len(maxs): return None, None, None, None
        return df.iloc[mins]['Close'], df.iloc[maxs]['Close'], mins, maxs
    except: return None, None, None, None

def analyze_market_regime():
    print(f">>> Fetching {MAIN_INDEX}...")
    df = get_data(MAIN_INDEX, "1d")
    if df is None: return 0, "NO_DATA", 0
    today = df.iloc[-1]
    return 0.0, "NORMAL", safe_float(today['Close'])

def scan_smt_for_set(set_key, timeframe, market_status, market_change):
    # --- FILTRELEME MANTIGI BURADA ---
    # Hesaplama yapilir AMA gondermeden once kontrol edilir.
    
    config = SMT_CONFIG[set_key]
    is_cluster = config.get("type") == "cluster"
    header = "🌅 **OPENING RANGE**" if is_opening_range() else "⚡ **INTRADAY**"
    
    # 1. Once sinyal var mi diye tum hesaplamalari yapalim (Backend Process)
    msg = None # Mesaj olusursa buraya dolacak

    # --- CLUSTER MODE ---
    if is_cluster:
        peers = config["peers"]
        data = {}
        for p in peers:
            df = get_data(p, timeframe)
            if df is None: continue
            trend_bias = check_trend_bias(df)
            l, h, l_idx, h_idx = find_swings(df, 2 if timeframe=="5m" else 3)
            if l is not None:
                atr = calculate_atr(df)
                data[p] = {"df":df, "l":l, "h":h, "l_idx":l_idx, "h_idx":h_idx, "atr":atr, "last":len(df)-1, "c":safe_float(df['Close'].iloc[-1]), "bias":trend_bias}
        
        if len(data) >= 2:
            for s1, s2 in combinations(data.keys(), 2):
                d1, d2 = data[s1], data[s2]
                
                # SHORT SIGNAL
                if (d1["last"]-d1["h_idx"][-1] <= FRESHNESS_LIMIT) and (d2["last"]-d2["h_idx"][-1] <= FRESHNESS_LIMIT):
                    leader = s1 if d1["h"].iloc[-1] > d1["h"].iloc[-2] and d2["h"].iloc[-1] < d2["h"].iloc[-2] else \
                             s2 if d2["h"].iloc[-1] > d2["h"].iloc[-2] and d1["h"].iloc[-1] < d1["h"].iloc[-2] else None
                    if leader:
                        laggard = s2 if leader == s1 else s1
                        main = data[leader]
                        hurst = calculate_hurst(main["df"])
                        markov = calculate_markov_prob(main["df"])
                        fvg = find_nearest_fvg(main["df"], "SHORT")
                        
                        msg = (f"{header} ({timeframe})\n{config['name']} ({s1} vs {s2})\n"
                               f"🚨 **ACTION: SHORT** 📉\n"
                               f"💪 Strong: {leader} | 🛑 Weak: {laggard}\n"
                               f"🧠 {markov} | {hurst}\n🧲 FVG: {fvg}")

                # LONG SIGNAL
                if (d1["last"]-d1["l_idx"][-1] <= FRESHNESS_LIMIT) and (d2["last"]-d2["l_idx"][-1] <= FRESHNESS_LIMIT):
                    leader = s1 if d1["l"].iloc[-1] < d1["l"].iloc[-2] and d2["l"].iloc[-1] > d2["l"].iloc[-2] else \
                             s2 if d2["l"].iloc[-1] < d2["l"].iloc[-2] and d1["l"].iloc[-1] > d1["l"].iloc[-2] else None
                    if leader:
                        laggard = s2 if leader == s1 else s1
                        main = data[leader]
                        hurst = calculate_hurst(main["df"])
                        markov = calculate_markov_prob(main["df"])
                        fvg = find_nearest_fvg(main["df"], "LONG")
                        
                        msg = (f"{header} ({timeframe})\n{config['name']} ({s1} vs {s2})\n"
                               f"🚨 **ACTION: LONG** 🚀\n"
                               f"📉 Sweeping: {leader} | 🛡️ Holding: {laggard}\n"
                               f"🧠 {markov} | {hurst}\n🧲 FVG: {fvg}")

    # --- STANDARD MODE ---
    else: 
        ref = config["ref"]
        df = get_data(ref, timeframe)
        if df is not None:
            l, h, l_idx, h_idx = find_swings(df, 2 if timeframe=="5m" else 3)
            if l is not None:
                atr = calculate_atr(df)
                rsi = calculate_rsi(df['Close'])
                data_ref = {"l":l, "h":h, "l_idx":l_idx, "h_idx":h_idx, "c":safe_float(df['Close'].iloc[-1]), "last":len(df)-1}
                comps = config["comps"]
                divs_short, divs_long = [], []
                
                for c in comps:
                    df_c = get_data(c, timeframe)
                    if df_c is not None:
                        lc, hc, _, _ = find_swings(df_c, 2 if timeframe=="5m" else 3)
                        if lc is not None:
                            if data_ref["h"].iloc[-1] > data_ref["h"].iloc[-2] and hc.iloc[-1] < hc.iloc[-2]: divs_short.append(c)
                            if data_ref["l"].iloc[-1] < data_ref["l"].iloc[-2] and lc.iloc[-1] > lc.iloc[-2]: divs_long.append(c)

                hurst = calculate_hurst(df)
                markov = calculate_markov_prob(df)
                
                # SHORT
                if divs_short and (data_ref["last"] - data_ref["h_idx"][-1] <= FRESHNESS_LIMIT):
                    fvg = find_nearest_fvg(df, "SHORT")
                    msg = (f"{header} ({timeframe})\n⚡ **{config['name']} SHORT**\n"
                           f"🚨 **ACTION: SHORT** 📉\n"
                           f"🛑 Divergence: {', '.join(divs_short)}\n"
                           f"🧠 {markov} | {hurst}\n🧲 FVG: {fvg}")

                # LONG
                if divs_long and (data_ref["last"] - data_ref["l_idx"][-1] <= FRESHNESS_LIMIT):
                    fvg = find_nearest_fvg(df, "LONG")
                    msg = (f"{header} ({timeframe})\n⚡ **{config['name']} LONG**\n"
                           f"🚨 **ACTION: LONG** 🚀\n"
                           f"🛑 Divergence: {', '.join(divs_long)}\n"
                           f"🧠 {markov} | {hurst}\n🧲 FVG: {fvg}")

    # --- KRITIK NOKTA: FILTRE KONTROLU ---
    if msg:
        # Eger Filtre 'ALL' ise GONDER.
        # Eger Filtre secili timeframe ile AYNI ise GONDER.
        # Degilse, gonderme (ama hesaplama yapildi, sistem calisiyor).
        if CURRENT_FILTER == "ALL" or CURRENT_FILTER == timeframe:
            send_telegram(msg)
        else:
            print(f">>> Sinyal bulundu ({timeframe}) ama filtre ({CURRENT_FILTER}) engelledi.")

if __name__ == "__main__":
    try:
        print(">>> Bot Started...")
        send_telegram("🖥️ **SYSTEM ONLINE**\nLoading Control Panel...")
        send_control_panel() # Baslangicta menuyu gonder
        
        while True:
            # 1. Once piyasa durumunu al
            m_pct, m_stat, m_prc = analyze_market_regime()
            
            # 2. Döngü sırasında stratejileri tara
            if m_stat != "NO_DATA":
                strats = list(SMT_CONFIG.keys())
                
                # OPENING RANGE (Ozel)
                if is_opening_range():
                    for s in strats:
                        scan_smt_for_set(s, TF_MICRO, m_stat, m_pct)
                        check_updates() # Her islem arasi buton kontrolu yap
                
                # NORMAL SCAN
                for s in strats:
                    for tf in [TF_SCALP, TF_SWING]:
                        scan_smt_for_set(s, tf, m_stat, m_pct)
                        check_updates() # Islem aralarinda dinle
            
            # 3. Bekleme Süresi (AMA DINLEYEREK)
            # Normalde time.sleep(60) yapardik. 
            # Şimdi 60 saniye boyunca her saniye buton kontrolu yapiyoruz.
            print(">>> Waiting next cycle...")
            for _ in range(60): # 1 dakikalik bekleme
                check_updates()
                time.sleep(1)
                
    except Exception as e:
        print(f"CRITICAL: {e}")
        traceback.print_exc()
        sys.exit(1)
