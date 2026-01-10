import sys
import traceback

# --- HATA YAKALAYICI BAŞLANGICI ---
try:
    print(">>> Bot Başlatılıyor... Kütüphaneler Yükleniyor...")
    import os
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import requests
    from scipy.signal import argrelextrema
    from datetime import datetime, time
    import pytz
    print(">>> Kütüphaneler Başarıyla Yüklendi.")

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
        if not TELEGRAM_TOKEN or not CHAT_ID: 
            print("!!! UYARI: Telegram Token veya Chat ID eksik! Mesaj atılamıyor.")
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try: 
            response = requests.post(url, data=data)
            if response.status_code != 200:
                print(f"!!! Telegram API Hatası: {response.text}")
            else:
                print(">>> Mesaj başarıyla Telegram'a iletildi.")
        except Exception as e: 
            print(f"!!! Telegram Bağlantı Hatası: {e}")

    # --- ZAMAN KONTROLÜ ---
    def get_ny_time():
        ny_tz = pytz.timezone('America/New_York')
        return datetime.now(ny_tz)

    def is_micro_scalp_time():
        """ 09:30 - 11:30 NY Saati """
        now_ny = get_ny_time().time()
        return time(9, 30) <= now_ny <= time(11, 30)

    # --- YENİ: SİSTEM OKEY MESAJI ---
    def send_system_ok_message():
        now = get_ny_time()
        msg = (f"🟢 **SİSTEM OKEY** 🟢\n"
               f"🕒 NY Saati: `{now.strftime('%H:%M')}`\n"
               f"✅ Bot: Çalışıyor\n"
               f"📡 Bağlantı: Başarılı")
        send_telegram(msg)

    # --- YARDIMCI: GÜVENLİ VERİ DÖNÜŞTÜRME ---
    def safe_float(val):
        """Pandas Series veya tekil değeri float'a çevirir"""
        try:
            if isinstance(val, pd.Series):
                return float(val.iloc[0])
            return float(val)
        except:
            return 0.0

    # --- BÖLÜM 1: 1 YILLIK VERİ İLE İSTATİSTİK ANALİZİ ---
    def analyze_market_regime():
        print(f">>> {MAIN_INDEX} Verisi Çekiliyor...")
        df = yf.download(MAIN_INDEX, period=STATS_DATA_PERIOD, interval="1d", progress=False)
        if len(df) < 10: 
            print("!!! Yetersiz Günlük Veri.")
            return 0, "VERI_YOK", 0
        
        df['Up_Move'] = (df['High'] - df['Open']) / df['Open']
        df['Down_Move'] = (df['Open'] - df['Low']) / df['Open']
        
        swing_up = np.percentile(df['Up_Move'].dropna(), SWING_PERCENTILE)
        swing_down = np.percentile(df['Down_Move'].dropna(), SWING_PERCENTILE)
        scalp_up = np.percentile(df['Up_Move'].dropna(), SCALP_PERCENTILE)
        scalp_down = np.percentile(df['Down_Move'].dropna(), SCALP_PERCENTILE)
        
        today = df.iloc[-1]
        
        # --- KRİTİK DÜZELTME BURADA YAPILDI ---
        # yfinance bazen Series döndürdüğü için safe_float ile sayıya zorluyoruz
        close_p = safe_float(today['Close'])
        open_p = safe_float(today['Open'])
        
        if open_p == 0: change_pct = 0
        else: change_pct = (close_p - open_p) / open_p
        # ---------------------------------------
        
        status = "NORMAL"
        if change_pct > 0:
            if change_pct > swing_up: status = "SWING_SHORT_ZONE"
            elif change_pct > scalp_up: status = "SCALP_SHORT_ZONE"
        elif change_pct < 0:
            down = abs(change_pct)
            if down > swing_down: status = "SWING_LONG_ZONE"
            elif down > scalp_down: status = "SCALP_LONG_ZONE"
            
        print(f">>> Piyasa Durumu: {status} (%{change_pct*100:.2f})")
        return change_pct, status, close_p

    # --- BÖLÜM 2: SMT TARAYICI ---
    def get_data(symbol, interval):
        p = "1d" if interval == "5m" else "5d"
        return yf.download(symbol, period=p, interval=interval, progress=False)

    def find_swings(df, order):
        if df is None or len(df) < 10: return None, None
        
        # NumPy array'e çevirerek işlem yap (Hata önleyici)
        close_vals = df['Close'].values.flatten() # Düzleştir
        
        mins = argrelextrema(close_vals, np.less_equal, order=order)[0]
        maxs = argrelextrema(close_vals, np.greater_equal, order=order)[0]
        
        if len(mins) == 0 or len(maxs) == 0: return None, None
        
        return df.iloc[mins]['Close'], df.iloc[maxs]['Close']

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
        if df_ref is None or len(df_ref) < 10: return
        
        lows, highs = find_swings(df_ref, order)
        if lows is None or len(lows) < 2 or len(highs) < 2: return

        # Değerleri güvenli sayıya çevir
        try:
            data_store["REF"] = {
                "L_new": safe_float(lows.iloc[-1]), "L_old": safe_float(lows.iloc[-2]),
                "H_new": safe_float(highs.iloc[-1]), "H_old": safe_float(highs.iloc[-2]),
                "Price": safe_float(df_ref['Close'].iloc[-1])
            }
        except: return

        # Karşılaştırma
        for sym in comp_tickers:
            df_c = get_data(sym, timeframe)
            if df_c is None: continue
            l, h = find_swings(df_c, order)
            if l is not None and len(l) >= 2 and len(h) >= 2:
                try:
                    data_store[sym] = {
                        "L_new": safe_float(l.iloc[-1]), "L_old": safe_float(l.iloc[-2]),
                        "H_new": safe_float(h.iloc[-1]), "H_old": safe_float(h.iloc[-2])
                    }
                except: continue
            else:
                return 

        msg = ""
        current_price = data_store["REF"]["Price"]

        # BEARISH SMT
        if data_store["REF"]["H_new"] > data_store["REF"]["H_old"]:
            divergences = []
            for sym in comp_tickers:
                if sym in data_store and data_store[sym]["H_new"] < data_store[sym]["H_old"]:
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
                if sym in data_store and data_store[sym]["L_new"] > data_store[sym]["L_old"]:
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
            print(f">>> Sinyal Bulundu: {strategy_name} ({timeframe})")
            send_telegram(msg)

    # --- ANA ÇALIŞTIRMA BLOĞU ---
    if __name__ == "__main__":
        print(">>> Bot Çalışma Döngüsü Başladı...")
        
        # 1. HER ÇALIŞMADA BİLDİRİM AT (Sistem Okey)
        send_system_ok_message()
        
        # 2. İSTATİSTİK
        m_change, m_status, m_price = analyze_market_regime()
        if m_status == "VERI_YOK":
            print("!!! Veri alınamadığı için analiz atlanıyor.")
        else:
            strategies = ["SET_1", "SET_2", "SET_3"]
            
            # 3. TARAMA
            # MİKRO SCALP (09:30-11:30)
            if is_micro_scalp_time():
                print(">>> Saat Uygun: Mikro Scalp (5m) Taranıyor...")
                for strat in strategies:
                    try: scan_smt_for_set(strat, TF_MICRO, m_status, m_change)
                    except Exception as e: print(f"Hata {strat} 5m: {e}")
            else:
                print(">>> Mikro Scalp Saati Değil. (5m Atlanıyor)")

            # GENEL TARAMA
            print(">>> Genel Tarama (15m & 1h) Başlıyor...")
            for strat in strategies:
                try: scan_smt_for_set(strat, TF_SCALP, m_status, m_change)
                except Exception as e: print(f"Hata {strat} 15m: {e}")
                
                try: scan_smt_for_set(strat, TF_SWING, m_status, m_change)
                except Exception as e: print(f"Hata {strat} 1h: {e}")

        print(">>> Bot Döngüsü Başarıyla Tamamlandı.")

# --- HATA YAKALAYICI SONU ---
except Exception as e:
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("CRITICAL ERROR: KOD ÇÖKTÜ!")
    print(f"HATA MESAJI: {e}")
    traceback.print_exc()
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    sys.exit(1)
