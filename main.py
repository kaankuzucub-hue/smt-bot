import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.signal import argrelextrema
from datetime import datetime, time
import pytz

# --- AYARLAR ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# --- KESİN KURAL: VERİ AYARLARI ---
STATS_DATA_PERIOD = "1y" 
MAIN_INDEX = "TQQQ"

# --- SMT SETLERİ ---
SMT_CONFIG = {
    "SET_1": {"name": "🔥 TQQQ TRIO", "ref": "TQQQ", "comps": ["SOXL", "NVDA"]},
    "SET_2": {"name": "⚖️ TQQQ-SEMI DUO", "ref": "TQQQ", "comps": ["SOXL"]},
    "SET_3": {"name": "💾 CHIP GIANTS", "ref": "AVGO", "comps": ["MU", "NVDA"]}
}

# Zaman Dilimleri
TF_MICRO = "5m"
TF_SCALP = "15m"
TF_SWING = "1h"

# --- İSTATİSTİKSEL EŞİKLER ---
SCALP_PERCENTILE = 75  
SWING_PERCENTILE = 90  

def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, data=data)
    except: pass

# --- ZAMAN KONTROLÜ ---
def get_ny_time():
    ny_tz = pytz.timezone('America/New_York')
    return datetime.now(ny_tz)

def is_micro_scalp_time():
    """ 09:30 - 11:30 NY Saati """
    now_ny = get_ny_time().time()
    return time(9, 30) <= now_ny <= time(11, 30)

def is_market_open():
    """ Haftasonu kontrolü ve Piyasa saati """
    now = get_ny_time()
    # Haftasonu ise (5=Cmt, 6=Paz) False dön
    if now.weekday() >= 5: return False
    # Saat kontrolü (09:30 - 16:00)
    return time(9, 30) <= now.time() <= time(16, 00)

# --- YENİ: GÜNLÜK İYİ GÜNLER MESAJI (KALP ATIŞI) ---
def daily_weekend_check():
    now = get_ny_time()
    
    # Sadece Haftasonları (Cumartesi veya Pazar)
    if now.weekday() >= 5:
        # Sadece Sabah 09:00 - 09:05 arasında (Günde 1 kez)
        # GitHub Actions 5 dakikada bir çalıştığı için bu aralığı mutlaka yakalar.
        if now.hour == 9 and 0 <= now.minute < 5:
            msg = (f"☀️ **MUTLU HAFTASONLARI!** ☀️\n\n"
                   f"🤖 **Durum:** Bot Aktif ve Çalışıyor.\n"
                   f"💤 **Piyasa:** Kapalı (Haftasonu Modu).\n"
                   f"🔋 **Sistem:** Sorunsuz.\n\n"
                   f"Pazartesi açılışta görüşmek üzere! 👋")
            send_telegram(msg)
            print("Günlük haftasonu mesajı gönderildi.")

# --- BÖLÜM 1: 1 YILLIK VERİ İLE İSTATİSTİK ANALİZİ ---
def analyze_market_regime():
    try:
        df = yf.download(MAIN_INDEX, period=STATS_DATA_PERIOD, interval="1d", progress=False)
        if len(df) < 200: return 0, "NORMAL"
        
        df['Up_Move'] = (df['High'] - df['Open']) / df['Open']
        df['Down_Move'] = (df['Open'] - df['Low']) / df['Open']
        
        swing_up = np.percentile(df['Up_Move'].dropna(), SWING_PERCENTILE)
        swing_down = np.percentile(df['Down_Move'].dropna(), SWING_PERCENTILE)
        scalp_up = np.percentile(df['Up_Move'].dropna(), SCALP_PERCENTILE)
        scalp_down = np.percentile(df['Down_Move'].dropna(), SCALP_PERCENTILE)
        
        today = df.iloc[-1]
        change_pct = (today['Close'] - today['Open']) / today['Open']
        
        status = "NORMAL"
        if change_pct > 0:
            if change_pct > swing_up: status = "SWING_SHORT_ZONE"
            elif change_pct > scalp_up: status = "SCALP_SHORT_ZONE"
        elif change_pct < 0:
            down = abs(change_pct)
            if down > swing_down: status = "SWING_LONG_ZONE"
            elif down > scalp_down: status = "SCALP_LONG_ZONE"
            
        return change_pct, status, today['Close']
    except:
        return 0, "HATA", 0

# --- BÖLÜM 2: SMT TARAYICI ---
def get_data(symbol, interval):
    try:
        p = "1d" if interval == "5m" else "5d"
        return yf.download(symbol, period=p, interval=interval, progress=False)
    except:
        return None

def find_swings(df, order):
    if df is None or len(df) < 10: return None, None
    df['min'] = df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=order)[0]]['Close']
    df['max'] = df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=order)[0]]['Close']
    return df['min'].dropna(), df['max'].dropna()

def scan_smt_for_set(set_key, timeframe, market_status, market_change):
    config = SMT_CONFIG[set_key]
    ref_ticker = config["ref"]
    comp_tickers = config["comps"]
    strategy_name = config["name"]
    
    if timeframe == "5m": order = 1
    elif timeframe == "15m": order = 2
    else: order = 3

    data_store = {}
    
    # Referans
    df_ref = get_data(ref_ticker, timeframe)
    lows, highs = find_swings(df_ref, order)
    if lows is None or len(lows) < 2 or len(highs) < 2: return

    data_store["REF"] = {
        "L_new": lows.iloc[-1], "L_old": lows.iloc[-2],
        "H_new": highs.iloc[-1], "H_old": highs.iloc[-2],
        "Price": df_ref['Close'].iloc[-1]
    }

    # Karşılaştırma
    for sym in comp_tickers:
        df_c = get_data(sym, timeframe)
        l, h = find_swings(df_c, order)
        if l is not None and len(l) >= 2 and len(h) >= 2:
            data_store[sym] = {
                "L_new": l.iloc[-1], "L_old": l.iloc[-2],
                "H_new": h.iloc[-1], "H_old": h.iloc[-2]
            }
        else:
            return 

    msg = ""
    current_price = data_store["REF"]["Price"]

    # BEARISH SMT
    if data_store["REF"]["H_new"] > data_store["REF"]["H_old"]:
        divergences = []
        for sym in comp_tickers:
            if data_store[sym]["H_new"] < data_store[sym]["H_old"]:
                divergences.append(sym)
        
        if divergences:
            icon = "🔬" if timeframe == "5m" else "⚡" if timeframe == "15m" else "🚨"
            comment = "Nötr (Teknik)"
            if "SHORT_ZONE" in market_status: comment = "🔥 GÜÇLÜ FIRSAT (İstatistik Onaylı)"
            elif "LONG_ZONE" in market_status: comment = "⚠️ TERS YÖN (Riskli)"

            msg = (f"{icon} **{strategy_name} SHORT ({timeframe})**\n\n"
                   f"📉 **Lider:** {ref_ticker} Yükseldi.\n"
                   f"🛑 **Onaylamayan:** {', '.join(divergences)}\n"
                   f"🌍 **Bölge:** {market_status} (%{market_change*100:.2f})\n"
                   f"🧠 **Yorum:** {comment}\n"
                   f"Fiyat: {current_price:.2f}")

    # BULLISH SMT
    elif data_store["REF"]["L_new"] < data_store["REF"]["L_old"]:
        divergences = []
        for sym in comp_tickers:
            if data_store[sym]["L_new"] > data_store[sym]["L_old"]:
                divergences.append(sym)
        
        if divergences:
            icon = "🔬" if timeframe == "5m" else "⚡" if timeframe == "15m" else "🚨"
            comment = "Nötr (Teknik)"
            if "LONG_ZONE" in market_status: comment = "🔥 GÜÇLÜ FIRSAT (İstatistik Onaylı)"
            elif "SHORT_ZONE" in market_status: comment = "⚠️ TERS YÖN (Riskli)"

            msg = (f"{icon} **{strategy_name} LONG ({timeframe})**\n\n"
                   f"📈 **Lider:** {ref_ticker} Düştü.\n"
                   f"💪 **Tutunan:** {', '.join(divergences)}\n"
                   f"🌍 **Bölge:** {market_status} (%{market_change*100:.2f})\n"
                   f"🧠 **Yorum:** {comment}\n"
                   f"Fiyat: {current_price:.2f}")

    if msg:
        send_telegram(msg)

# --- ANA ÇALIŞTIRMA BLOĞU ---
if __name__ == "__main__":
    
    # 1. GÜNLÜK KALP ATIŞI (HAFTASONU KONTROLÜ)
    daily_weekend_check()
    
    # 2. PİYASA AÇIK MI?
    # Eğer haftasonuysa veya piyasa kapalıysa aşağıyı çalıştırma, kaynak tüketme.
    if not is_market_open():
        # EĞER MANUEL OLARAK TEST EDİYORSAN (GitHub'dan butona bastıysan)
        # Buraya küçük bir test mesajı ekleyebiliriz ki çalıştığını gör.
        # Ama otomatik döngüde spam yapmasın diye kapalı tutuyoruz.
        print("Piyasa Kapalı. Dinlenme Modu.")
    
    else:
        # Piyasa Açıksa Analize Başla
        m_change, m_status, m_price = analyze_market_regime()
        strategies = ["SET_1", "SET_2", "SET_3"]
        
        # MİKRO SCALP (09:30-11:30)
        if is_micro_scalp_time():
            for strat in strategies:
                try: scan_smt_for_set(strat, TF_MICRO, m_status, m_change)
                except: pass

        # GENEL TARAMA
        for strat in strategies:
            try: scan_smt_for_set(strat, TF_SCALP, m_status, m_change)
            except: pass
            try: scan_smt_for_set(strat, TF_SWING, m_status, m_change)
            except: pass
