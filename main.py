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

# --- GLOBAL STATE ---
CURRENT_FILTER = "ALL"
LAST_UPDATE_ID = 0
SENT_MESSAGES = {} 

# --- SETTINGS ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# --- MONEY MANAGEMENT ---
ACCOUNT_SIZE = 100000
RISK_AMOUNT = 1000
REWARD_RATIO = 2.0

# --- DATA RULES ---
STATS_DATA_PERIOD = "1y" # Kayıtlı tercihin: Her zaman 1 yıllık veri

# --- SMT CONFIGURATION ---
SMT_CONFIG = {
    "SET_1": {"type": "standard", "name": "🔥 TQQQ TRIO", "ref": "TQQQ", "comps": ["SOXL", "NVDA"]},
    "SET_2": {"type": "standard", "name": "⚖️ TQQQ SEMI DUO", "ref": "TQQQ", "comps": ["SOXL"]},
    "SET_3": {"type": "cluster", "name": "⚔️ CHIP WARS (Matrix)", "peers": ["NVDA", "AVGO", "MU"]},
    "SET_4": {"type": "standard", "name": "🏥 SECTOR X-RAY", "ref": "TQQQ", "comps": ["XLK", "XLC", "XLY", "SMH"]},
    "SET_X9191": {"type": "cluster", "name": "👽 PROTOCOL X-9191", "peers": ["TQQQ", "XLK", "SMH"]}
}

TF_MICRO, TF_SCALP, TF_SWING = "5m", "15m", "1h"
FRESHNESS_LIMIT = 5

# --- TELEGRAM CORE ---

def send_telegram(message, cache_key=None):
    """Spam engelleme mekanizmalı mesaj gönderici"""
    global SENT_MESSAGES
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    
    # Eğer bu key ile aynı mesaj zaten atılmışsa, fonksiyonu burada bitir.
    if cache_key and SENT_MESSAGES.get(cache_key) == message:
        return 

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data, timeout=10)
        if cache_key:
            SENT_MESSAGES[cache_key] = message 
    except: pass

def send_control_panel():
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    keyboard = {
        "inline_keyboard": [
            [{"text": "⚡ 5m", "callback_data": "5m"}, {"text": "🎯 15m", "callback_data": "15m"}, {"text": "⏳ 1h", "callback_data": "1h"}],
            [{"text": "👁️ SHOW ALL (Reset)", "callback_data": "ALL"}],
            [{"text": "📊 Status", "callback_data": "STATUS"}]
        ]
    }
    msg_text = f"🎛️ **TQQQ QUANT PANEL**\nFilter: **{CURRENT_FILTER}**"
    data = {"chat_id": CHAT_ID, "text": msg_text, "parse_mode": "Markdown", "reply_markup": json.dumps(keyboard)}
    try: requests.post(url, data=data, timeout=10)
    except: pass

def check_updates():
    global LAST_UPDATE_ID, CURRENT_FILTER
    if not TELEGRAM_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    try:
        resp = requests.get(url, params={"offset": LAST_UPDATE_ID + 1, "timeout": 1}, timeout=2).json()
        if "result" in resp:
            for update in resp["result"]:
                LAST_UPDATE_ID = update["update_id"]
                if "message" in update and "text" in update["message"]:
                    if update["message"]["text"] == "/menu": send_control_panel()
                elif "callback_query" in update:
                    selection = update["callback_query"]["data"]
                    if selection in ["5m", "15m", "1h", "ALL"]:
                        CURRENT_FILTER = selection
                        # FOTOĞRAFTAKİ HATANIN ÇÖZÜMÜ: Filtre mesajına cache_key eklendi.
                        send_telegram(f"⚙️ **FILTER:** {CURRENT_FILTER}", cache_key="filter_status")
                    elif selection == "STATUS":
                        send_telegram(f"✅ Active\nFilter: {CURRENT_FILTER}\nSignals in Cache: {len(SENT_MESSAGES)}", cache_key="status_msg")
                    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery", 
                                  data={"callback_query_id": update["callback_query"]["id"]})
    except: pass

# --- MATH & QUANT ENGINE (KORUNDU) ---

def get_data(symbol, interval):
    try:
        df = yf.download(symbol, period=STATS_DATA_PERIOD, interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

def find_swings(df, order):
    try:
        c = df['Close'].values.flatten()
        l_idx = argrelextrema(c, np.less_equal, order=order)[0]
        h_idx = argrelextrema(c, np.greater_equal, order=order)[0]
        return df.iloc[l_idx]['Close'], df.iloc[h_idx]['Close'], l_idx, h_idx
    except: return None, None, None, None

def calculate_hurst(df):
    try:
        ts = df['Close'].values.flatten()
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        h = np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0
        return f"🌊 Trending ({h:.2f})" if h > 0.55 else f"🪃 MeanRev ({h:.2f})"
    except: return "N/A"

def calculate_markov_prob(df):
    try:
        states = (df['Close'].pct_change().dropna().tail(100) > 0).astype(int)
        tm = np.zeros((2, 2))
        for i in range(len(states)-1): tm[states.iloc[i]][states.iloc[i+1]] += 1
        curr = states.iloc[-1]
        p_bull = (tm[curr][1] / np.sum(tm[curr])) * 100
        return f"🐂 Bull %{p_bull:.0f}" if p_bull > 60 else f"🐻 Bear %{100-p_bull:.0f}" if p_bull < 40 else "⚖️ Neutral"
    except: return "N/A"

def calculate_atr(df):
    tr = pd.concat([(df['High']-df['Low']), (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(14).mean().iloc[-1])

# --- SCANNER ---

def scan_smt_for_set(set_key, timeframe):
    if CURRENT_FILTER != "ALL" and CURRENT_FILTER != timeframe: return

    config = SMT_CONFIG[set_key]
    is_cluster = config.get("type") == "cluster"
    
    if is_cluster:
        peers = config["peers"]
        data = {}
        for p in peers:
            df = get_data(p, timeframe)
            if df is None: continue
            l, h, l_idx, h_idx = find_swings(df, 2 if timeframe=="5m" else 3)
            if len(l_idx) > 0 and len(h_idx) > 0:
                data[p] = {"df":df, "l":l, "h":h, "l_idx":l_idx, "h_idx":h_idx, "last":len(df)-1, "c":float(df['Close'].iloc[-1]), "atr":calculate_atr(df)}
        
        if len(data) < 2: return
        for s1, s2 in combinations(data.keys(), 2):
            d1, d2 = data[s1], data[s2]
            cache_key = f"sig_{set_key}_{s1}_{s2}_{timeframe}"
            
            # SHORT CHECK
            if (d1["last"]-d1["h_idx"][-1] <= FRESHNESS_LIMIT) and (d2["last"]-d2["h_idx"][-1] <= FRESHNESS_LIMIT):
                leader = s1 if d1["h"].iloc[-1] > d1["h"].iloc[-2] and d2["h"].iloc[-1] < d2["h"].iloc[-2] else s2 if d2["h"].iloc[-1] > d2["h"].iloc[-2] and d1["h"].iloc[-1] < d1["h"].iloc[-2] else None
                if leader:
                    m = data[leader]
                    msg = (f"🚨 **{config['name']} SHORT**\n💪 Leader: {leader}\n⏱️ TF: {timeframe}\n"
                           f"🧠 {calculate_markov_prob(m['df'])} | {calculate_hurst(m['df'])}\n"
                           f"🛑 SL: {m['c'] + 1.5*m['atr']:.2f} | 💰 TP: {m['c'] - 3.0*m['atr']:.2f}")
                    send_telegram(msg, cache_key=cache_key)
                elif cache_key in SENT_MESSAGES: del SENT_MESSAGES[cache_key]
            
            # LONG CHECK
            elif (d1["last"]-d1["l_idx"][-1] <= FRESHNESS_LIMIT) and (d2["last"]-d2["l_idx"][-1] <= FRESHNESS_LIMIT):
                leader = s1 if d1["l"].iloc[-1] < d1["l"].iloc[-2] and d2["l"].iloc[-1] > d2["l"].iloc[-2] else s2 if d2["l"].iloc[-1] < d2["l"].iloc[-2] and d1["l"].iloc[-1] > d1["l"].iloc[-2] else None
                if leader:
                    m = data[leader]
                    msg = (f"🚀 **{config['name']} LONG**\n📉 Sweeping: {leader}\n⏱️ TF: {timeframe}\n"
                           f"🧠 {calculate_markov_prob(m['df'])} | {calculate_hurst(m['df'])}\n"
                           f"🛑 SL: {m['c'] - 1.5*m['atr']:.2f} | 💰 TP: {m['c'] + 3.0*m['atr']:.2f}")
                    send_telegram(msg, cache_key=cache_key)
                elif cache_key in SENT_MESSAGES: del SENT_MESSAGES[cache_key]
            else:
                if cache_key in SENT_MESSAGES: del SENT_MESSAGES[cache_key]

# --- MAIN ---

if __name__ == "__main__":
    send_telegram("🟢 **SYSTEM OPERATIONAL**\n👽 Module: X-9191 ACTIVE", cache_key="boot_msg")
    send_control_panel()
    
    while True:
        # Önce Telegram komutlarını oku (Buton tıklamaları vb.)
        check_updates()
        
        # Stratejileri tara
        for tf in [TF_MICRO, TF_SCALP, TF_SWING]:
            for s in ["SET_1", "SET_2", "SET_3", "SET_4", "SET_X9191"]:
                try: 
                    scan_smt_for_set(s, tf)
                    # Her strateji arasında kısa bir check_updates ki butonlar takılmasın
                    check_updates() 
                except: pass
        
        # Döngü sonu bekleme
        time.sleep(2)
