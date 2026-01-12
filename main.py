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

    # --- MONEY MANAGEMENT ---
    ACCOUNT_SIZE = 100000 
    RISK_AMOUNT = 1000    
    REWARD_RATIO = 2.0    

    # --- DATA RULES ---
    STATS_DATA_PERIOD = "1y" 
    MAIN_INDEX = "TQQQ"

    # --- SMT CONFIGURATION ---
    SMT_CONFIG = {
        "SET_1": { "type": "standard", "name": "🔥 TQQQ TRIO", "ref": "TQQQ", "comps": ["SOXL", "NVDA"] },
        "SET_2": { "type": "standard", "name": "⚖️ TQQQ SEMI DUO", "ref": "TQQQ", "comps": ["SOXL"] },
        "SET_3": { "type": "cluster", "name": "⚔️ CHIP WARS (Matrix)", "peers": ["NVDA", "AVGO", "MU"] },
        "SET_4": { "type": "standard", "name": "🏥 SECTOR X-RAY", "ref": "TQQQ", "comps": ["XLK", "XLC", "XLY", "SMH"] }
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
               f"🧠 Brain: LEVEL 4 (Instit. Flow & Stats)")
        send_telegram(msg)

    # --- HELPERS ---
    def safe_float(val):
        try:
            if isinstance(val, pd.Series): return float(val.iloc[0])
            return float(val)
        except: return 0.0

    # ==========================================
    # 🧠 LAYER 3: QUANT RISK (ATR, FVG, VIX)
    # ==========================================
    def calculate_atr(df, period=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        return safe_float(atr)
    
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def get_vix_sentiment():
        try:
            vix = yf.download("^VIX", period="2d", interval="1d", progress=False)
            if len(vix) < 1: return "N/A"
            current_vix = safe_float(vix['Close'].iloc[-1])
            if current_vix > 25: return f"🌪️ **EXTREME ({current_vix:.0f})**"
            elif current_vix > 18: return f"⚠️ **High ({current_vix:.0f})**"
            else: return f"🌊 **Safe ({current_vix:.0f})**"
        except: return "N/A"

    def find_nearest_fvg(df, direction):
        try:
            closes = df['Close'].values; highs = df['High'].values; lows = df['Low'].values
            fvg_price = 0; fvg_dist = 99999; current_price = closes[-1]
            for i in range(len(closes)-2, len(closes)-20, -1):
                if direction == "SHORT":
                    if lows[i] > highs[i+2]:
                        gap_mid = (lows[i] + highs[i+2]) / 2
                        if gap_mid < current_price:
                            dist = current_price - gap_mid
                            if dist < fvg_dist: fvg_dist, fvg_price = dist, gap_mid
                elif direction == "LONG":
                    if highs[i] < lows[i+2]:
                        gap_mid = (highs[i] + lows[i+2]) / 2
                        if gap_mid > current_price:
                            dist = gap_mid - current_price
                            if dist < fvg_dist: fvg_dist, fvg_price = dist, gap_mid
            return f"${fvg_price:.2f}" if fvg_price != 0 else "None"
        except: return "Err"

    # ==========================================
    # 🏦 LAYER 4: INSTITUTIONAL FLOW & STATS
    # ==========================================

    # 1. Z-SCORE (Standard Deviation from Mean)
    def calculate_z_score(df, period=20):
        # Fiyatın ortalamadan kaç standart sapma saptığını bulur.
        # +2.0 üzeri aşırı pahalı, -2.0 altı aşırı ucuzdur.
        try:
            closes = df['Close']
            sma = closes.rolling(window=period).mean()
            std = closes.rolling(window=period).std()
            z_score = (closes.iloc[-1] - sma.iloc[-1]) / std.iloc[-1]
            
            tag = ""
            if z_score > 3.0: tag = "🔥 **EXTREME OVERBOUGHT (+3σ)**"
            elif z_score > 2.0: tag = "⚠️ **Overextended (+2σ)**"
            elif z_score < -3.0: tag = "💎 **EXTREME OVERSOLD (-3σ)**"
            elif z_score < -2.0: tag = "♻️ **Undervalued (-2σ)**"
            else: tag = f"Neutral ({z_score:.2f}σ)"
            return tag
        except: return "N/A"

    # 2. MFI (Money Flow Index - Hacim Destekli RSI)
    def calculate_mfi(df, period=14):
        # RSI gibidir ama Hacmi de hesaba katar.
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            
            positive_flow = [0] * len(money_flow)
            negative_flow = [0] * len(money_flow)
            
            for i in range(1, len(money_flow)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow[i] = money_flow.iloc[i]
                elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                    negative_flow[i] = money_flow.iloc[i]
            
            pos_mf = pd.Series(positive_flow).rolling(period).sum()
            neg_mf = pd.Series(negative_flow).rolling(period).sum()
            
            mfi = 100 - (100 / (1 + (pos_mf / neg_mf)))
            return safe_float(mfi.iloc[-1])
        except: return 50.0

    # 3. VWAP DEVIATION (Kurumsal Maliyet Analizi)
    def check_vwap_status(df):
        # Basitleştirilmiş Rolling VWAP (Son 20 mumluk hacim ağırlıklı ortalama)
        # Eğer fiyat VWAP'ten çok uzaksa "Mean Reversion" beklenir.
        try:
            v = df['Volume'].values
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            tp_v = tp * v
            
            cum_v = pd.Series(v).rolling(20).sum()
            cum_tp_v = pd.Series(tp_v).rolling(20).sum()
            vwap = cum_tp_v / cum_v
            
            current = df['Close'].iloc[-1]
            vwap_val = vwap.iloc[-1]
            
            dist_pct = ((current - vwap_val) / vwap_val) * 100
            
            if dist_pct > 2.0: return f"Expensive (+{dist_pct:.1f}% vs VWAP)"
            elif dist_pct < -2.0: return f"Cheap ({dist_pct:.1f}% vs VWAP)"
            else: return "Fair Value (At VWAP)"
        except: return "N/A"

    # ==========================================
    # 🔙 BACKTEST ENGINE 
    # ==========================================
    def check_past_trade(df, entry_idx, direction, atr):
        try:
            entry_price = safe_float(df['Close'].iloc[entry_idx])
            if direction == "SHORT":
                stop_loss = entry_price + (atr * 1.5)
                take_profit = entry_price - (atr * 3.0)
            else: 
                stop_loss = entry_price - (atr * 1.5)
                take_profit = entry_price + (atr * 3.0)
            
            outcome = "⏳ **PENDING**"
            pnl_str = "$0"
            future_data = df.iloc[entry_idx+1:]
            
            for i in range(len(future_data)):
                high = safe_float(future_data['High'].iloc[i])
                low = safe_float(future_data['Low'].iloc[i])
                if direction == "SHORT":
                    if low <= take_profit: 
                        outcome = "🏆 **WIN**"; pnl_str = f"+${RISK_AMOUNT * REWARD_RATIO:,.0f}"; break
                    if high >= stop_loss: 
                        outcome = "❌ **LOSS**"; pnl_str = f"-${RISK_AMOUNT:,.0f}"; break
                else: 
                    if high >= take_profit: 
                        outcome = "🏆 **WIN**"; pnl_str = f"+${RISK_AMOUNT * REWARD_RATIO:,.0f}"; break
                    if low <= stop_loss: 
                        outcome = "❌ **LOSS**"; pnl_str = f"-${RISK_AMOUNT:,.0f}"; break
            return f"{outcome} ({pnl_str})"
        except: return "N/A"

    def generate_trade_plan(price, direction, atr):
        if direction == "SHORT": stop = price + (atr * 1.5); target = price - (atr * 3.0)
        else: stop = price - (atr * 1.5); target = price + (atr * 3.0)
        return f"🛑 Stop: {stop:.2f}\n💰 Target: {target:.2f} (1:2)"

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
        change_pct = (close_p - open_p) / open_p if open_p != 0 else 0
        status = "NORMAL"
        if change_pct > 0:
            if change_pct > swing_up: status = "SWING_SHORT_ZONE"
            elif change_pct > scalp_up: status = "SCALP_SHORT_ZONE"
        elif change_pct < 0: down = abs(change_pct); status = "SWING_LONG_ZONE" if down > swing_down else "SCALP_LONG_ZONE" if down > scalp_down else status
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
        time_header = "🌅 **OPENING RANGE SNIPER**" if is_opening_range() else "⚡ **INTRADAY SCAN**"

        # --- LOGIC A: CLUSTER MODE ---
        if strategy_type == "cluster":
            peers = config["peers"]
            peer_data = {}
            vix_msg = get_vix_sentiment()
            
            for p in peers:
                df = get_data(p, timeframe)
                if df is None: continue
                l, h, l_idx, h_idx = find_swings(df, order)
                if l is not None:
                    atr_val = calculate_atr(df)
                    # LAYER 4 CALCS
                    z_tag = calculate_z_score(df)
                    mfi_val = calculate_mfi(df)
                    vwap_st = check_vwap_status(df)
                    
                    peer_data[p] = {
                        "L_new": safe_float(l.iloc[-1]), "L_old": safe_float(l.iloc[-2]),
                        "H_new": safe_float(h.iloc[-1]), "H_old": safe_float(h.iloc[-2]),
                        "H_idx": h_idx[-1], "L_idx": l_idx[-1],
                        "Close": safe_float(df['Close'].iloc[-1]),
                        "ATR": atr_val, "Last_Bar": len(df) - 1, "DF": df,
                        "Z": z_tag, "MFI": mfi_val, "VWAP": vwap_st
                    }
            
            if len(peer_data) < 2: return

            for s1, s2 in combinations(peer_data.keys(), 2):
                d1, d2 = peer_data[s1], peer_data[s2]
                
                # SHORT
                is_bearish = False
                leader, laggard = "", ""
                if d1["H_new"] > d1["H_old"] and d2["H_new"] < d2["H_old"]:
                    if (d1["Last_Bar"] - d1["H_idx"] <= FRESHNESS_LIMIT): is_bearish, leader, laggard = True, s1, s2
                elif d2["H_new"] > d2["H_old"] and d1["H_new"] < d1["H_old"]:
                    if (d2["Last_Bar"] - d2["H_idx"] <= FRESHNESS_LIMIT): is_bearish, leader, laggard = True, s2, s1
                
                if is_bearish:
                    main = peer_data[leader]
                    trade_plan = generate_trade_plan(main["Close"], "SHORT", main["ATR"])
                    fvg = find_nearest_fvg(main["DF"], "SHORT")
                    past_res = check_past_trade(main["DF"], main["H_idx"][-1], "SHORT", main["ATR"])

                    msg = (f"{time_header}\n⚔️ **CHIP WAR ({s1} vs {s2})**\n\n"
                           f"🚨 **ACTION: SHORT** 📉\n------------------------\n"
                           f"💪 **Strong:** {leader} (HH)\n🛑 **Weak:** {laggard} (LH)\n"
                           f"⏱️ **TF:** {timeframe}\n🧠 **Reason:** Bearish Divergence\n"
                           f"========================\n"
                           f"📊 **QUANT LAYER:**\n🌪️ **VIX:** {vix_msg}\n🧲 **FVG:** {fvg}\n"
                           f"🔙 **PREV TRADE:** {past_res}\n"
                           f"========================\n"
                           f"🏦 **INSTITUTIONAL FLOW:**\n"
                           f"📏 **Z-Score:** {main['Z']}\n"
                           f"💸 **MFI:** {main['MFI']:.0f} (Vol Momentum)\n"
                           f"⚖️ **VWAP:** {main['VWAP']}\n"
                           f"{trade_plan}")
                    send_telegram(msg)

                # LONG
                is_bullish = False
                if d1["L_new"] < d1["L_old"] and d2["L_new"] > d2["L_old"]:
                    if (d1["Last_Bar"] - d1["L_idx"] <= FRESHNESS_LIMIT): is_bullish, leader, laggard = True, s1, s2
                elif d2["L_new"] < d2["L_old"] and d1["L_new"] > d1["L_old"]:
                    if (d2["Last_Bar"] - d2["L_idx"] <= FRESHNESS_LIMIT): is_bullish, leader, laggard = True, s2, s1
                
                if is_bullish:
                    main = peer_data[leader]
                    trade_plan = generate_trade_plan(main["Close"], "LONG", main["ATR"])
                    fvg = find_nearest_fvg(main["DF"], "LONG")
                    past_res = check_past_trade(main["DF"], main["L_idx"][-1], "LONG", main["ATR"])

                    msg = (f"{time_header}\n⚔️ **CHIP WAR ({s1} vs {s2})**\n\n"
                           f"🚨 **ACTION: LONG** 🚀\n------------------------\n"
                           f"📉 **Sweeping:** {leader} (LL)\n🛡️ **Holding:** {laggard} (HL)\n"
                           f"⏱️ **TF:** {timeframe}\n🧠 **Reason:** Bullish Divergence\n"
                           f"========================\n"
                           f"📊 **QUANT LAYER:**\n🌪️ **VIX:** {vix_msg}\n🧲 **FVG:** {fvg}\n"
                           f"🔙 **PREV TRADE:** {past_res}\n"
                           f"========================\n"
                           f"🏦 **INSTITUTIONAL FLOW:**\n"
                           f"📏 **Z-Score:** {main['Z']}\n"
                           f"💸 **MFI:** {main['MFI']:.0f} (Vol Momentum)\n"
                           f"⚖️ **VWAP:** {main['VWAP']}\n"
                           f"{trade_plan}")
                    send_telegram(msg)

        # --- LOGIC B: STANDARD MODE ---
        else:
            ref_ticker = config["ref"]
            comp_tickers = config["comps"]
            df_ref = get_data(ref_ticker, timeframe)
            if df_ref is None: return
            
            rsi_series = calculate_rsi(df_ref['Close'])
            atr_val = calculate_atr(df_ref)
            vix_msg = get_vix_sentiment()
            
            # LAYER 4 CALCS
            z_tag = calculate_z_score(df_ref)
            mfi_val = calculate_mfi(df_ref)
            vwap_st = check_vwap_status(df_ref)

            l, h, l_idx, h_idx = find_swings(df_ref, order)
            if l is None: return
            last_candle_idx = len(df_ref) - 1
            
            try:
                rsi_new_h, rsi_old_h = safe_float(rsi_series.iloc[h_idx[-1]]), safe_float(rsi_series.iloc[h_idx[-2]])
                rsi_new_l, rsi_old_l = safe_float(rsi_series.iloc[l_idx[-1]]), safe_float(rsi_series.iloc[l_idx[-2]])
            except: rsi_new_h=rsi_old_h=rsi_new_l=rsi_old_l=50

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

            # SHORT
            if data_store["REF"]["H_new"] > data_store["REF"]["H_old"]:
                if (last_candle_idx - data_store["REF"]["H_idx"] <= FRESHNESS_LIMIT):
                    divs = [s for s in comp_tickers if s in data_store and data_store[s]["H_new"] < data_store[s]["H_old"]]
                    if divs:
                        is_rsi_div = (rsi_new_h < rsi_old_h)
                        header = f"💣 **RSI + SMT SETUP**" if is_rsi_div else f"⚡ **STANDARD SMT**"
                        rsi_txt = f"📉 **Div:** {rsi_old_h:.0f}->{rsi_new_h:.0f}" if is_rsi_div else f"RSI: {rsi_new_h:.0f}"
                        trade_plan = generate_trade_plan(data_store["REF"]["Price"], "SHORT", atr_val)
                        fvg = find_nearest_fvg(df_ref, "SHORT")
                        past_res = check_past_trade(df_ref, h_idx[-1], "SHORT", atr_val)

                        msg = (f"{time_header}\n{header}\n\n"
                               f"🚨 **ACTION: SHORT** 📉\n------------------------\n"
                               f"📉 **Leader:** {ref_ticker}\n🛑 **Laggard:** {', '.join(divs)}\n"
                               f"🔋 **Momentum:** {rsi_txt}\n🧠 **Reason:** Bearish Setup\n"
                               f"========================\n"
                               f"📊 **QUANT LAYER:**\n🌪️ **VIX:** {vix_msg}\n🧲 **FVG:** {fvg}\n"
                               f"🔙 **PREV TRADE:** {past_res}\n"
                               f"========================\n"
                               f"🏦 **INSTITUTIONAL FLOW:**\n"
                               f"📏 **Z-Score:** {z_tag}\n"
                               f"💸 **MFI:** {mfi_val:.0f} (Vol Momentum)\n"
                               f"⚖️ **VWAP:** {vwap_st}\n"
                               f"{trade_plan}")
                        send_telegram(msg)

            # LONG
            elif data_store["REF"]["L_new"] < data_store["REF"]["L_old"]:
                if (last_candle_idx - data_store["REF"]["L_idx"] <= FRESHNESS_LIMIT):
                    divs = [s for s in comp_tickers if s in data_store and data_store[s]["L_new"] > data_store[s]["L_old"]]
                    if divs:
                        is_rsi_div = (rsi_new_l > rsi_old_l)
                        header = f"🚀 **RSI + SMT SETUP**" if is_rsi_div else f"⚡ **STANDARD SMT**"
                        rsi_txt = f"📈 **Div:** {rsi_old_l:.0f}->{rsi_new_l:.0f}" if is_rsi_div else f"RSI: {rsi_new_l:.0f}"
                        trade_plan = generate_trade_plan(data_store["REF"]["Price"], "LONG", atr_val)
                        fvg = find_nearest_fvg(df_ref, "LONG")
                        past_res = check_past_trade(df_ref, l_idx[-1], "LONG", atr_val)

                        msg = (f"{time_header}\n{header}\n\n"
                               f"🚨 **ACTION: LONG** 🚀\n------------------------\n"
                               f"📈 **Leader:** {ref_ticker}\n💪 **Holding:** {', '.join(divs)}\n"
                               f"🔋 **Momentum:** {rsi_txt}\n🧠 **Reason:** Bullish Setup\n"
                               f"========================\n"
                               f"📊 **QUANT LAYER:**\n🌪️ **VIX:** {vix_msg}\n🧲 **FVG:** {fvg}\n"
                               f"🔙 **PREV TRADE:** {past_res}\n"
                               f"========================\n"
                               f"🏦 **INSTITUTIONAL FLOW:**\n"
                               f"📏 **Z-Score:** {z_tag}\n"
                               f"💸 **MFI:** {mfi_val:.0f} (Vol Momentum)\n"
                               f"⚖️ **VWAP:** {vwap_st}\n"
                               f"{trade_plan}")
                        send_telegram(msg)

    if __name__ == "__main__":
        print(">>> Loop Started...")
        send_system_ok_message()
        m_change, m_status, m_price = analyze_market_regime()
        if m_status != "NO_DATA":
            strats = ["SET_1", "SET_2", "SET_3", "SET_4"]
            if is_opening_range():
                for s in strats: 
                    try: scan_smt_for_set(s, TF_MICRO, m_status, m_change)
                    except: pass
            for s in strats:
                try: scan_smt_for_set(s, TF_SCALP, m_status, m_change)
                except: pass
                try: scan_smt_for_set(s, TF_SWING, m_status, m_change)
                except: pass
        print(">>> Loop Finished.")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
