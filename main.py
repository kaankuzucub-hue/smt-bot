import sys
import traceback

# --- ERROR HANDLER START ---
try:
    print(">>> Bot Starting... Loading Libraries...")
    import os
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import requests
    from scipy.signal import argrelextrema
    from datetime import datetime, time
    import pytz
    from itertools import combinations
    print(">>> Libraries Loaded Successfully.")

    # --- SETTINGS ---
    TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
    CHAT_ID = os.environ.get("CHAT_ID")

    # --- DATA RULES ---
    STATS_DATA_PERIOD = "1y" 
    MAIN_INDEX = "TQQQ"

    # --- SMT CONFIGURATION ---
    SMT_CONFIG = {
        # SET 1: TQQQ TRIO (Klasik)
        "SET_1": {
            "type": "standard",
            "name": "🔥 TQQQ TRIO",
            "ref": "TQQQ", 
            "comps": ["SOXL", "NVDA"] 
        },
        # SET 2: TQQQ DUO (Klasik)
        "SET_2": {
            "type": "standard",
            "name": "⚖️ TQQQ SEMI DUO",
            "ref": "TQQQ",
            "comps": ["SOXL"]
        },
        # SET 3: CHIP CLUSTER (Hisse Matrix)
        "SET_3": {
            "type": "cluster",
            "name": "⚔️ CHIP WARS (Matrix)",
            "peers": ["NVDA", "AVGO", "MU"] 
        },
        # SET 4: SECTOR X-RAY (YENİ - ETF ANALİZİ)
        # TQQQ Yükselirken Alt Sektörler (XLK, XLC, XLY, SMH) Destekliyor mu?
        "SET_4": {
            "type": "standard", # Lider: TQQQ
            "name": "🏥 SECTOR X-RAY",
            "ref": "TQQQ",
            # Teknoloji, İletişim, Tüketim, Yarı İletken
            "comps": ["XLK", "XLC", "XLY", "SMH"] 
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

    def send_telegram(message):
        if not TELEGRAM_TOKEN or not CHAT_ID: 
            print("!!! WARNING: Token or Chat ID missing.")
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try: requests.post(url, data=data)
        except Exception as e: print(f"!!! Telegram Error: {e}")

    # --- TIME CONTROL ---
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
               f"📡 Mode: SECTOR X-RAY ADDED")
        send_telegram(msg)

    # --- HELPERS ---
    def safe_float(val):
        try:
            if isinstance(val, pd.Series): return float(val.iloc[0])
            return float(val)
        except: return 0.0

    # --- RSI CALCULATOR ---
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # --- PART 1: MARKET STATS ---
    def analyze_market_regime():
        print(f">>> Fetching {MAIN_INDEX} Data...")
        df = yf.download(MAIN_INDEX, period=STATS_DATA_PERIOD, interval="1d", progress=False)
        if len(df) < 10: return 0, "NO_DATA", 0
        
        df['Up_Move'] = (df['High'] - df['Open']) / df['Open']
        df['Down_Move'] = (df['Open'] - df['Low']) / df['Open']
        
        swing_up = np.percentile(df['Up_Move'].dropna(), SWING_PERCENTILE)
        swing_down = np.percentile(df['Down_Move'].dropna(), SWING_PERCENTILE)
        scalp_up = np.percentile(df['Up_Move'].dropna(), SCALP_PERCENTILE)
        scalp_down = np.percentile(df['Down_Move'].dropna(), SCALP_PERCENTILE)
        
        today = df.iloc[-1]
        close_p = safe_float(today['Close'])
        open_p = safe_float(today['Open'])
        if open_p == 0: change_pct = 0
        else: change_pct = (close_p - open_p) / open_p
        
        status = "NORMAL"
        if change_pct > 0:
            if change_pct > swing_up: status = "SWING_SHORT_ZONE"
            elif change_pct > scalp_up: status = "SCALP_SHORT_ZONE"
        elif change_pct < 0:
            down = abs(change_pct)
            if down > swing_down: status = "SWING_LONG_ZONE"
            elif down > scalp_down: status = "SCALP_LONG_ZONE"
            
        print(f">>> Market Status: {status} ({change_pct*100:.2f}%)")
        return change_pct, status, close_p

    # --- PART 2: DATA & SWINGS ---
    def get_data(symbol, interval):
        p = "1d" if interval == "5m" else "5d"
        return yf.download(symbol, period=p, interval=interval, progress=False)

    def find_swings(df, order):
        if df is None or len(df) < 10: return None, None, None, None
        close_vals = df['Close'].values.flatten()
        mins_idx = argrelextrema(close_vals, np.less_equal, order=order)[0]
        maxs_idx = argrelextrema(close_vals, np.greater_equal, order=order)[0]
        if len(mins_idx) == 0 or len(maxs_idx) == 0: return None, None, None, None
        return df.iloc[mins_idx]['Close'], df.iloc[maxs_idx]['Close'], mins_idx, maxs_idx

    # --- PART 3: SCANNER ENGINE ---
    def scan_smt_for_set(set_key, timeframe, market_status, market_change):
        config = SMT_CONFIG[set_key]
        strategy_type = config.get("type", "standard")
        strategy_name = config["name"]
        
        if timeframe == "5m": order = 1
        elif timeframe == "15m": order = 2
        else: order = 3

        if is_opening_range(): time_header = "🌅 **OPENING RANGE SNIPER**"
        else: time_header = "⚡ **INTRADAY SCAN**"

        # --- LOGIC A: CLUSTER MODE (HİSSE MATRIX - AYNI) ---
        if strategy_type == "cluster":
            peers = config["peers"]
            peer_data = {}
            for p in peers:
                df = get_data(p, timeframe)
                if df is None: continue
                l, h, l_idx, h_idx = find_swings(df, order)
                if l is not None:
                    peer_data[p] = {
                        "L_new": safe_float(l.iloc[-1]), "L_old": safe_float(l.iloc[-2]),
                        "H_new": safe_float(h.iloc[-1]), "H_old": safe_float(h.iloc[-2]),
                        "H_idx": h_idx[-1], "L_idx": l_idx[-1],
                        "Last_Bar": len(df) - 1
                    }
            
            if len(peer_data) < 2: return

            for s1, s2 in combinations(peer_data.keys(), 2):
                d1, d2 = peer_data[s1], peer_data[s2]
                
                # Bearish
                if d1["H_new"] > d1["H_old"] and d2["H_new"] < d2["H_old"]:
                    if (d1["Last_Bar"] - d1["H_idx"] <= FRESHNESS_LIMIT):
                        msg = (f"{time_header}\n⚔️ **CHIP WAR ({s1} vs {s2})**\n\n"
                               f"🚨 **ACTION: SHORT** 📉\n"
                               f"------------------------\n"
                               f"💪 **Strong:** {s1} (HH)\n🛑 **Weak:** {s2} (LH)\n"
                               f"⏱️ **TF:** {timeframe}\n🧠 Divergence in Sector")
                        send_telegram(msg)
                elif d2["H_new"] > d2["H_old"] and d1["H_new"] < d1["H_old"]:
                    if (d2["Last_Bar"] - d2["H_idx"] <= FRESHNESS_LIMIT):
                        msg = (f"{time_header}\n⚔️ **CHIP WAR ({s2} vs {s1})**\n\n"
                               f"🚨 **ACTION: SHORT** 📉\n"
                               f"------------------------\n"
                               f"💪 **Strong:** {s2} (HH)\n🛑 **Weak:** {s1} (LH)\n"
                               f"⏱️ **TF:** {timeframe}\n🧠 Divergence in Sector")
                        send_telegram(msg)
                
                # Bullish
                if d1["L_new"] < d1["L_old"] and d2["L_new"] > d2["L_old"]:
                    if (d1["Last_Bar"] - d1["L_idx"] <= FRESHNESS_LIMIT):
                        msg = (f"{time_header}\n⚔️ **CHIP WAR ({s1} vs {s2})**\n\n"
                               f"🚨 **ACTION: LONG** 🚀\n"
                               f"------------------------\n"
                               f"📉 **Sweeping:** {s1} (LL)\n🛡️ **Holding:** {s2} (HL)\n"
                               f"⏱️ **TF:** {timeframe}\n🧠 Divergence in Sector")
                        send_telegram(msg)
                elif d2["L_new"] < d2["L_old"] and d1["L_new"] > d1["L_old"]:
                    if (d2["Last_Bar"] - d2["L_idx"] <= FRESHNESS_LIMIT):
                        msg = (f"{time_header}\n⚔️ **CHIP WAR ({s2} vs {s1})**\n\n"
                               f"🚨 **ACTION: LONG** 🚀\n"
                               f"------------------------\n"
                               f"📉 **Sweeping:** {s2} (LL)\n🛡️ **Holding:** {s1} (HL)\n"
                               f"⏱️ **TF:** {timeframe}\n🧠 Divergence in Sector")
                        send_telegram(msg)


        # --- LOGIC B: STANDARD MODE (TQQQ/SECTOR X-RAY) ---
        else:
            ref_ticker = config["ref"]
            comp_tickers = config["comps"]
            
            df_ref = get_data(ref_ticker, timeframe)
            if df_ref is None: return
            
            rsi_series = calculate_rsi(df_ref['Close'])
            
            l, h, l_idx, h_idx = find_swings(df_ref, order)
            if l is None: return

            last_candle_idx = len(df_ref) - 1
            
            # Get RSI at Swing Points
            try:
                rsi_new_high = safe_float(rsi_series.iloc[h_idx[-1]])
                rsi_old_high = safe_float(rsi_series.iloc[h_idx[-2]])
                rsi_new_low = safe_float(rsi_series.iloc[l_idx[-1]])
                rsi_old_low = safe_float(rsi_series.iloc[l_idx[-2]])
            except: 
                rsi_new_high = rsi_old_high = rsi_new_low = rsi_old_low = 50 

            data_store = {}
            try:
                data_store["REF"] = {
                    "L_new": safe_float(l.iloc[-1]), "L_old": safe_float(l.iloc[-2]),
                    "H_new": safe_float(h.iloc[-1]), "H_old": safe_float(h.iloc[-2]),
                    "Price": safe_float(df_ref['Close'].iloc[-1]),
                    "H_idx": h_idx[-1], "L_idx": l_idx[-1]
                }
            except: return

            for sym in comp_tickers:
                df_c = get_data(sym, timeframe)
                if df_c is None: continue
                l, h, _, _ = find_swings(df_c, order)
                if l is not None:
                    try:
                        data_store[sym] = {
                            "L_new": safe_float(l.iloc[-1]), "L_old": safe_float(l.iloc[-2]),
                            "H_new": safe_float(h.iloc[-1]), "H_old": safe_float(h.iloc[-2])
                        }
                    except: continue

            # --- BEARISH SMT CHECK ---
            if data_store["REF"]["H_new"] > data_store["REF"]["H_old"]:
                bars_ago = last_candle_idx - data_store["REF"]["H_idx"]
                if bars_ago <= FRESHNESS_LIMIT:
                    divs = []
                    for s in comp_tickers:
                        if s in data_store and data_store[s]["H_new"] < data_store[s]["H_old"]:
                            divs.append(s)
                    
                    if divs:
                        has_rsi_div = (rsi_new_high < rsi_old_high)
                        
                        if has_rsi_div:
                            final_header = f"💣 **RSI + SMT SETUP**"
                            rsi_msg = f"📉 **RSI Div:** {rsi_old_high:.0f} -> {rsi_new_high:.0f}"
                            comment = "🔥 **HIGH PROBABILITY**"
                        else:
                            final_header = f"⚡ **STANDARD SMT**"
                            rsi_msg = f"RSI: {rsi_new_high:.0f} (No Div)"
                            comment = "Asset Divergence"

                        msg = (f"{time_header}\n{final_header}\n\n"
                               f"🚨 **ACTION: SHORT** 📉\n"
                               f"------------------------\n"
                               f"📉 **Leader:** {ref_ticker} HH\n"
                               f"🛑 **Weak Sectors:** {', '.join(divs)}\n"
                               f"🔋 **Momentum:** {rsi_msg}\n"
                               f"🕯️ **Freshness:** {bars_ago} bars\n"
                               f"🧠 {comment}\nPrice: {data_store['REF']['Price']:.2f}")
                        send_telegram(msg)

            # --- BULLISH SMT CHECK ---
            elif data_store["REF"]["L_new"] < data_store["REF"]["L_old"]:
                bars_ago = last_candle_idx - data_store["REF"]["L_idx"]
                if bars_ago <= FRESHNESS_LIMIT:
                    divs = []
                    for s in comp_tickers:
                        if s in data_store and data_store[s]["L_new"] > data_store[s]["L_old"]:
                            divs.append(s)
                    
                    if divs:
                        has_rsi_div = (rsi_new_low > rsi_old_low)
                        
                        if has_rsi_div:
                            final_header = f"🚀 **RSI + SMT SETUP**"
                            rsi_msg = f"📈 **RSI Div:** {rsi_old_low:.0f} -> {rsi_new_low:.0f}"
                            comment = "🔥 **HIGH PROBABILITY**"
                        else:
                            final_header = f"⚡ **STANDARD SMT**"
                            rsi_msg = f"RSI: {rsi_new_low:.0f} (No Div)"
                            comment = "Asset Divergence"

                        msg = (f"{time_header}\n{final_header}\n\n"
                               f"🚨 **ACTION: LONG** 🚀\n"
                               f"------------------------\n"
                               f"📈 **Leader:** {ref_ticker} LL\n"
                               f"💪 **Holding Sectors:** {', '.join(divs)}\n"
                               f"🔋 **Momentum:** {rsi_msg}\n"
                               f"🕯️ **Freshness:** {bars_ago} bars\n"
                               f"🧠 {comment}\nPrice: {data_store['REF']['Price']:.2f}")
                        send_telegram(msg)

    # --- MAIN ---
    if __name__ == "__main__":
        print(">>> Execution Loop Started...")
        send_system_ok_message()
        m_change, m_status, m_price = analyze_market_regime()
        
        if m_status != "NO_DATA":
            strategies = ["SET_1", "SET_2", "SET_3", "SET_4"]
            
            if is_opening_range():
                print(">>> Opening Range Scan...")
                for strat in strategies:
                    try: scan_smt_for_set(strat, TF_MICRO, m_status, m_change)
                    except: pass
            
            print(">>> Standard Scan...")
            for strat in strategies:
                try: scan_smt_for_set(strat, TF_SCALP, m_status, m_change)
                except: pass
                try: scan_smt_for_set(strat, TF_SWING, m_status, m_change)
                except: pass
        print(">>> Finished.")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
