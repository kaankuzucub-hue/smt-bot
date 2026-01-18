#!/usr/bin/env python3
"""
🚀 ADVANCED SMT TRADING BOT v2.0
Single-file version - Ready for GitHub deployment
"""

import sys
import traceback
import os
import time
import threading
import sqlite3
import json
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import hashlib

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.signal import argrelextrema
import pytz
from itertools import combinations

# ==========================================
# 📊 CONFIGURATION
# ==========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# Money Management
ACCOUNT_SIZE = 100000
RISK_AMOUNT = 1000
REWARD_RATIO = 2.0

# Data Settings
STATS_DATA_PERIOD = "1y"
MAIN_INDEX = "TQQQ"

# SMT Configuration
SMT_CONFIG = {
    "SET_1": {"type": "standard", "name": "🔥 TQQQ TRIO", "ref": "TQQQ", "comps": ["SOXL", "NVDA"]},
    "SET_2": {"type": "standard", "name": "⚖️ TQQQ SEMI DUO", "ref": "TQQQ", "comps": ["SOXL"]},
    "SET_3": {"type": "cluster", "name": "⚔️ CHIP WARS", "peers": ["NVDA", "AVGO", "MU"]},
    "SET_4": {"type": "standard", "name": "🏥 SECTOR X-RAY", "ref": "TQQQ", "comps": ["XLK", "XLC", "XLY", "SMH"]},
    "SET_X9191": {"type": "cluster", "name": "👽 PROTOCOL X-9191", "peers": ["TQQQ", "XLK", "SMH"]}
}

# Timeframes
TIMEFRAMES = {
    "5m": {"name": "⚡ Micro (5m)", "period": "1d", "swing_order": 2},
    "15m": {"name": "🎯 Scalp (15m)", "period": "5d", "swing_order": 3},
    "1h": {"name": "📈 Swing (1h)", "period": "5d", "swing_order": 3}
}

# Thresholds
FRESHNESS_LIMIT = 5
SIGNAL_COOLDOWN = 3600  # 1 hour
SCAN_INTERVAL = 300  # 5 minutes

# ==========================================
# 🗄️ DATABASE SETUP
# ==========================================
DB_PATH = "trading_bot.db"

def init_database():
    """Initialize SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            strategy TEXT,
            timeframe TEXT,
            symbol TEXT,
            direction TEXT,
            entry_price REAL,
            stop_loss REAL,
            take_profit REAL,
            status TEXT DEFAULT 'ACTIVE',
            result TEXT,
            profit_loss REAL,
            signal_hash TEXT UNIQUE
        )''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            active_timeframes TEXT DEFAULT '["5m","15m","1h"]',
            notifications_enabled INTEGER DEFAULT 1,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized")
    except Exception as e:
        print(f"Database init error: {e}")

# ==========================================
# 📝 LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 📡 TELEGRAM FUNCTIONS
# ==========================================
def send_telegram(message, reply_markup=None):
    """Send message to Telegram"""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.warning("Telegram credentials missing")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    if reply_markup:
        data["reply_markup"] = json.dumps(reply_markup)
    
    try:
        response = requests.post(url, data=data, timeout=10)
        if response.status_code != 200:
            logger.error(f"Telegram error: {response.text}")
    except Exception as e:
        logger.error(f"Telegram send error: {e}")

def get_telegram_updates(offset=None):
    """Get updates from Telegram"""
    if not TELEGRAM_TOKEN:
        return []
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    params = {"offset": offset, "timeout": 30}
    
    try:
        response = requests.get(url, params=params, timeout=35)
        if response.status_code == 200:
            return response.json().get("result", [])
    except Exception as e:
        logger.error(f"Get updates error: {e}")
    return []

def send_main_menu():
    """Send interactive menu"""
    keyboard = {
        "keyboard": [
            [{"text": "⚡ 5m Only"}, {"text": "🎯 15m Only"}, {"text": "📈 1h Only"}],
            [{"text": "🌟 All Timeframes"}, {"text": "⏸️ Pause"}],
            [{"text": "📊 Statistics"}, {"text": "⚙️ Settings"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False
    }
    
    msg = (
        "🤖 **ADVANCED SMT BOT v2.0**\n\n"
        "⚡ **5m** - Micro scalping\n"
        "🎯 **15m** - Intraday\n"
        "📈 **1h** - Swing trading\n"
        "🌟 **All** - All timeframes\n\n"
        "📊 Statistics - View performance\n"
        "⚙️ Settings - Bot config\n"
        "⏸️ Pause - Stop signals"
    )
    
    send_telegram(msg, keyboard)

def get_user_preferences(user_id):
    """Get user preferences"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT active_timeframes, notifications_enabled FROM user_preferences WHERE user_id=?", (user_id,))
        row = c.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0]), bool(row[1])
        return ["5m", "15m", "1h"], True
    except:
        return ["5m", "15m", "1h"], True

def update_user_preferences(user_id, timeframes=None, notifications=None):
    """Update preferences"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        if timeframes is not None:
            tf_json = json.dumps(timeframes)
            c.execute("""INSERT INTO user_preferences (user_id, active_timeframes) 
                        VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET active_timeframes=?""",
                     (user_id, tf_json, tf_json))
        
        if notifications is not None:
            c.execute("""INSERT INTO user_preferences (user_id, notifications_enabled) 
                        VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET notifications_enabled=?""",
                     (user_id, int(notifications), int(notifications)))
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Update preferences error: {e}")

def get_statistics():
    """Generate statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("""
            SELECT 
                direction,
                COUNT(*) as total,
                SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(profit_loss) as total_pnl
            FROM signals
            GROUP BY direction
        """, conn)
        conn.close()
        
        if df.empty:
            return "📊 **No signals yet**\n\nStart monitoring!"
        
        msg = "📊 **PERFORMANCE STATS**\n\n"
        
        for _, row in df.iterrows():
            direction = row['direction']
            total = int(row['total'])
            wins = int(row['wins'])
            losses = int(row['losses'])
            pnl = row['total_pnl'] if pd.notna(row['total_pnl']) else 0
            
            win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
            icon = "🚀" if direction == "LONG" else "📉"
            
            msg += (
                f"{icon} **{direction}**\n"
                f"├─ Total: {total} | Wins: {wins} | Losses: {losses}\n"
                f"├─ Win Rate: {win_rate:.1f}%\n"
                f"└─ P&L: ${pnl:,.0f}\n\n"
            )
        
        return msg
    except Exception as e:
        return f"❌ Error: {e}"

# ==========================================
# 🧮 HELPER FUNCTIONS
# ==========================================
def get_ny_time():
    ny_tz = pytz.timezone('America/New_York')
    return datetime.now(ny_tz)

def is_opening_range():
    now_ny = get_ny_time().time()
    from datetime import time as dt_time
    return dt_time(9, 30) <= now_ny <= dt_time(11, 30)

def is_market_open():
    now_ny = get_ny_time()
    if now_ny.weekday() >= 5:
        return False
    now_time = now_ny.time()
    from datetime import time as dt_time
    return dt_time(9, 30) <= now_time <= dt_time(16, 0)

def safe_float(val):
    try:
        if isinstance(val, pd.Series):
            if val.empty:
                return 0.0
            return float(val.iloc[0])
        return float(val)
    except:
        return 0.0

def generate_signal_hash(strategy, timeframe, symbol, direction, entry_price):
    data = f"{strategy}_{timeframe}_{symbol}_{direction}_{entry_price:.2f}"
    return hashlib.md5(data.encode()).hexdigest()

def check_signal_cooldown(signal_hash):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT timestamp FROM signals WHERE signal_hash=? ORDER BY timestamp DESC LIMIT 1", (signal_hash,))
        row = c.fetchone()
        conn.close()
        
        if row:
            last_time = datetime.fromisoformat(row[0])
            elapsed = (datetime.now() - last_time).total_seconds()
            return elapsed < SIGNAL_COOLDOWN
        return False
    except:
        return False

def save_signal(strategy, timeframe, symbol, direction, entry_price, stop_loss, take_profit, signal_hash):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""INSERT INTO signals (strategy, timeframe, symbol, direction, entry_price, stop_loss, take_profit, signal_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
            (strategy, timeframe, symbol, direction, entry_price, stop_loss, take_profit, signal_hash))
        conn.commit()
        conn.close()
        logger.info(f"Signal saved: {strategy} {symbol} {direction}")
    except sqlite3.IntegrityError:
        logger.debug("Duplicate signal ignored")
    except Exception as e:
        logger.error(f"Save signal error: {e}")

# ==========================================
# 📈 TECHNICAL ANALYSIS
# ==========================================
def check_trend_bias(df):
    try:
        if len(df) < 200:
            return "NEUTRAL"
        ema200 = df['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
        current_price = safe_float(df['Close'].iloc[-1])
        return "BULLISH" if current_price > ema200 else "BEARISH"
    except:
        return "NEUTRAL"

def calculate_markov_prob(df):
    try:
        closes = df['Close'].pct_change().dropna().tail(100)
        if len(closes) < 10:
            return "N/A"
        states = (closes > 0).astype(int)
        trans_mat = np.zeros((2, 2))
        for i in range(len(states)-1):
            curr, next_s = states.iloc[i], states.iloc[i+1]
            trans_mat[curr][next_s] += 1
        current_state = states.iloc[-1]
        row_sum = np.sum(trans_mat[current_state])
        if row_sum == 0:
            return "N/A"
        prob_bull = (trans_mat[current_state][1] / row_sum) * 100
        if prob_bull > 60:
            return f"🐂 {prob_bull:.0f}%"
        elif prob_bull < 40:
            return f"🐻 {100-prob_bull:.0f}%"
        return f"⚖️ {prob_bull:.0f}%"
    except:
        return "N/A"

def calculate_z_score(df, period=20):
    try:
        closes = df['Close']
        if len(closes) < period:
            return "N/A"
        z = (closes.iloc[-1] - closes.rolling(period).mean().iloc[-1]) / closes.rolling(period).std().iloc[-1]
        if z > 3.0:
            return "🔥 EXTREME (+3σ)"
        elif z < -3.0:
            return "💎 EXTREME (-3σ)"
        return f"Neutral ({z:.1f}σ)"
    except:
        return "N/A"

def calculate_mfi(df, period=14):
    try:
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        mf = tp * df['Volume']
        pos = mf.where(tp.diff() > 0, 0).rolling(period).sum()
        neg = mf.where(tp.diff() < 0, 0).rolling(period).sum()
        if neg.iloc[-1] == 0:
            return 50.0
        return safe_float(100 - (100 / (1 + (pos.iloc[-1] / neg.iloc[-1]))))
    except:
        return 50.0

def calculate_atr(df, period=14):
    try:
        h_l = df['High'] - df['Low']
        h_c = (df['High'] - df['Close'].shift()).abs()
        l_c = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
        return safe_float(tr.rolling(period).mean().iloc[-1])
    except:
        return 0.0

def get_vix_sentiment():
    try:
        vix = yf.download("^VIX", period="2d", interval="1d", progress=False)
        if vix is None or len(vix) < 1:
            return "N/A"
        val = safe_float(vix['Close'].iloc[-1])
        return f"🌪️ {val:.0f}" if val > 25 else f"🌊 {val:.0f}"
    except:
        return "N/A"

def generate_trade_plan(price, direction, atr):
    if atr <= 0:
        return ""
    sl = price + 1.5 * atr if direction == "SHORT" else price - 1.5 * atr
    tp = price - 3.0 * atr if direction == "SHORT" else price + 3.0 * atr
    return f"🛑 Stop: {sl:.2f}\n💰 Target: {tp:.2f}"

# ==========================================
# 📊 DATA FETCHING
# ==========================================
def get_data(symbol, interval):
    try:
        period = TIMEFRAMES[interval]["period"]
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or len(df) < 2:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return None

def find_swings(df, order):
    if df is None or len(df) < 10:
        return None, None, None, None
    try:
        c = df['Close'].values.flatten()
        mins = argrelextrema(c, np.less_equal, order=order)[0]
        maxs = argrelextrema(c, np.greater_equal, order=order)[0]
        if not len(mins) or not len(maxs):
            return None, None, None, None
        return df.iloc[mins]['Close'], df.iloc[maxs]['Close'], mins, maxs
    except:
        return None, None, None, None

# ==========================================
# 🎯 CORE SCANNING LOGIC
# ==========================================
def scan_smt_for_set(set_key, timeframe, active_tfs):
    """Scan SMT divergence"""
    
    if timeframe not in active_tfs:
        return
    
    config = SMT_CONFIG[set_key]
    is_cluster = config.get("type") == "cluster"
    
    if is_cluster:
        peers = config["peers"]
        data = {}
        
        for p in peers:
            df = get_data(p, timeframe)
            if df is None:
                continue
            
            trend_bias = check_trend_bias(df)
            swing_order = TIMEFRAMES[timeframe]["swing_order"]
            l, h, l_idx, h_idx = find_swings(df, swing_order)
            
            if l is not None:
                atr = calculate_atr(df)
                data[p] = {
                    "df": df, "l": l, "h": h, "l_idx": l_idx, "h_idx": h_idx,
                    "atr": atr, "last": len(df) - 1, "c": safe_float(df['Close'].iloc[-1]),
                    "bias": trend_bias
                }
        
        if len(data) < 2:
            return
        
        for s1, s2 in combinations(data.keys(), 2):
            d1, d2 = data[s1], data[s2]
            
            # SHORT CHECK
            if (d1["last"] - d1["h_idx"][-1] <= FRESHNESS_LIMIT) and \
               (d2["last"] - d2["h_idx"][-1] <= FRESHNESS_LIMIT):
                
                leader = s1 if d1["h"].iloc[-1] > d1["h"].iloc[-2] and d2["h"].iloc[-1] < d2["h"].iloc[-2] else \
                         s2 if d2["h"].iloc[-1] > d2["h"].iloc[-2] and d1["h"].iloc[-1] < d1["h"].iloc[-2] else None
                
                if leader:
                    laggard = s2 if leader == s1 else s1
                    main = data[leader]
                    
                    signal_hash = generate_signal_hash(set_key, timeframe, leader, "SHORT", main['c'])
                    
                    if check_signal_cooldown(signal_hash):
                        continue
                    
                    sl = main['c'] + 1.5 * main['atr']
                    tp = main['c'] - 3.0 * main['atr']
                    
                    save_signal(config['name'], timeframe, leader, "SHORT", main['c'], sl, tp, signal_hash)
                    
                    trend_status = "⚠️ RISKY" if main["bias"] == "BULLISH" else "✅ Aligned"
                    
                    msg = (
                        f"{'🌅 OPENING' if is_opening_range() else '⚡ INTRADAY'}\n"
                        f"{config['name']} ({s1} vs {s2})\n\n"
                        f"🚨 **SHORT** 📉 ({trend_status})\n"
                        f"────────────────────\n"
                        f"💪 Leader: {leader} (HH)\n"
                        f"🛑 Weak: {laggard} (LH)\n"
                        f"⏱️ TF: {timeframe} | 💵 ${main['c']:.2f}\n"
                        f"────────────────────\n"
                        f"🧠 {calculate_markov_prob(main['df'])}\n"
                        f"📏 {calculate_z_score(main['df'])}\n"
                        f"💸 MFI: {calculate_mfi(main['df']):.0f} | VIX: {get_vix_sentiment()}\n"
                        f"────────────────────\n"
                        f"{generate_trade_plan(main['c'], 'SHORT', main['atr'])}"
                    )
                    
                    send_telegram(msg)
                    logger.info(f"SHORT signal: {leader} {timeframe}")
            
            # LONG CHECK
            if (d1["last"] - d1["l_idx"][-1] <= FRESHNESS_LIMIT) and \
               (d2["last"] - d2["l_idx"][-1] <= FRESHNESS_LIMIT):
                
                leader = s1 if d1["l"].iloc[-1] < d1["l"].iloc[-2] and d2["l"].iloc[-1] > d2["l"].iloc[-2] else \
                         s2 if d2["l"].iloc[-1] < d2["l"].iloc[-2] and d1["l"].iloc[-1] > d1["l"].iloc[-2] else None
                
                if leader:
                    laggard = s2 if leader == s1 else s1
                    main = data[leader]
                    
                    signal_hash = generate_signal_hash(set_key, timeframe, leader, "LONG", main['c'])
                    
                    if check_signal_cooldown(signal_hash):
                        continue
                    
                    sl = main['c'] - 1.5 * main['atr']
                    tp = main['c'] + 3.0 * main['atr']
                    
                    save_signal(config['name'], timeframe, leader, "LONG", main['c'], sl, tp, signal_hash)
                    
                    trend_status = "⚠️ RISKY" if main["bias"] == "BEARISH" else "✅ Aligned"
                    
                    msg = (
                        f"{'🌅 OPENING' if is_opening_range() else '⚡ INTRADAY'}\n"
                        f"{config['name']} ({s1} vs {s2})\n\n"
                        f"🚨 **LONG** 🚀 ({trend_status})\n"
                        f"────────────────────\n"
                        f"📉 Sweep: {leader} (LL)\n"
                        f"🛡️ Hold: {laggard} (HL)\n"
                        f"⏱️ TF: {timeframe} | 💵 ${main['c']:.2f}\n"
                        f"────────────────────\n"
                        f"🧠 {calculate_markov_prob(main['df'])}\n"
                        f"📏 {calculate_z_score(main['df'])}\n"
                        f"💸 MFI: {calculate_mfi(main['df']):.0f} | VIX: {get_vix_sentiment()}\n"
                        f"────────────────────\n"
                        f"{generate_trade_plan(main['c'], 'LONG', main['atr'])}"
                    )
                    
                    send_telegram(msg)
                    logger.info(f"LONG signal: {leader} {timeframe}")
    
    else:  # Standard mode (same logic as cluster but comparing ref vs comps)
        ref = config["ref"]
        df = get_data(ref, timeframe)
        if df is None:
            return
        
        trend_bias = check_trend_bias(df)
        swing_order = TIMEFRAMES[timeframe]["swing_order"]
        l, h, l_idx, h_idx = find_swings(df, swing_order)
        
        if l is None:
            return
        
        atr = calculate_atr(df)
        data_ref = {
            "l": l, "h": h, "l_idx": l_idx, "h_idx": h_idx,
            "c": safe_float(df['Close'].iloc[-1]), "last": len(df) - 1
        }
        
        comps = config["comps"]
        divs_short, divs_long = [], []
        
        for c in comps:
            df_c = get_data(c, timeframe)
            if df_c is None:
                continue
            lc, hc, _, _ = find_swings(df_c, swing_order)
            if lc is not None:
                if data_ref["h"].iloc[-1] > data_ref["h"].iloc[-2] and hc.iloc[-1] < hc.iloc[-2]:
                    divs_short.append(c)
                if data_ref["l"].iloc[-1] < data_ref["l"].iloc[-2] and lc.iloc[-1] > lc.iloc[-2]:
                    divs_long.append(c)
        
        # SHORT
        if divs_short and (data_ref["last"] - data_ref["h_idx"][-1] <= FRESHNESS_LIMIT):
            signal_hash = generate_signal_hash(set_key, timeframe, ref, "SHORT", data_ref['c'])
            
            if not check_signal_cooldown(signal_hash):
                sl = data_ref['c'] + 1.5 * atr
                tp = data_ref['c'] - 3.0 * atr
                
                save_signal(config['name'], timeframe, ref, "SHORT", data_ref['c'], sl, tp, signal_hash)
                
                trend_status = "⚠️ RISKY" if trend_bias == "BULLISH" else "✅ Aligned"
                
                msg = (
                    f"{'🌅 OPENING' if is_opening_range() else '⚡ INTRADAY'}\n"
                    f"{config['name']}\n\n"
                    f"🚨 **SHORT** 📉 ({trend_status})\n"
                    f"────────────────────\n"
                    f"📉 Leader: {ref} (HH)\n"
                    f"🛑 Laggard: {', '.join(divs_short)}\n"
                    f"⏱️ TF: {timeframe} | 💵 ${data_ref['c']:.2f}\n"
                    f"────────────────────\n"
                    f"🧠 {calculate_markov_prob(df)}\n"
                    f"📏 {calculate_z_score(df)}\n"
                    f"💸 MFI: {calculate_mfi(df):.0f} | VIX: {get_vix_sentiment()}\n"
                    f"────────────────────\n"
                    f"{generate_trade_plan(data_ref['c'], 'SHORT', atr)}"
                )
                
                send_telegram(msg)
                logger.info(f"SHORT signal: {ref} {timeframe}")
        
        # LONG
        if divs_long and (data_ref["last"] - data_ref["l_idx"][-1] <= FRESHNESS_LIMIT):
            signal_hash = generate_signal_hash(set_key, timeframe, ref, "LONG", data_ref['c'])
            
            if not check_signal_cooldown(signal_hash):
                sl = data_ref['c'] - 1.5 * atr
                tp = data_ref['c'] + 3.0 * atr
                
                save_signal(config['name'], timeframe, ref, "LONG", data_ref['c'], sl, tp, signal_hash)
                
                trend_status = "⚠️ RISKY" if trend_bias == "BEARISH" else "✅ Aligned"
                
                msg = (
                    f"{'🌅 OPENING' if is_opening_range() else '⚡ INTRADAY'}\n"
                    f"{config['name']}\n\n"
                    f"🚨 **LONG** 🚀 ({trend_status})\n"
                    f"────────────────────\n"
                    f"📈 Leader: {ref} (LL)\n"
                    f"🛡️ Holding: {', '.join(divs_long)}\n"
                    f"⏱️ TF: {timeframe} | 💵 ${data_ref['c']:.2f}\n"
                    f"────────────────────\n"
                    f"🧠 {calculate_markov_prob(df)}\n"
                    f"📏 {calculate_z_score(df)}\n"
                    f"💸 MFI: {calculate_mfi(df):.0f} | VIX: {get_vix_sentiment()}\n"
                    f"────────────────────\n"
                    f"{generate_trade_plan(data_ref['c'], 'LONG', atr)}"
                )
                
                send_telegram(msg)
                logger.info(f"LONG signal: {ref} {timeframe}")

# ==========================================
# 🔄 CONTINUOUS MONITORING
# ==========================================
class TradingBotMonitor:
    """Main bot class"""
    
    def __init__(self):
        self.running = False
        self.last_update_id = None
        
    def start(self):
        self.running = True
        logger.info("🚀 Bot started")
        
        monitor_thread = threading.Thread(target=self.monitor_markets, daemon=True)
        monitor_thread.start()
        
        command_thread = threading.Thread(target=self.handle_commands, daemon=True)
        command_thread.start()
        
        send_main_menu()
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        self.running = False
        logger.info("🛑 Bot stopped")
    
    def monitor_markets(self):
        logger.info("📡 Market monitoring started")
        
        while self.running:
            try:
                if not is_market_open():
                    logger.debug("Market closed")
                    time.sleep(300)
                    continue
                
                logger.info("🔍 Scanning...")
                
                active_tfs, notifications = get_user_preferences(CHAT_ID)
                
                if not notifications:
                    time.sleep(SCAN_INTERVAL)
                    continue
                
                strategies = ["SET_1", "SET_2", "SET_3", "SET_4", "SET_X9191"]
                
                for strategy in strategies:
                    for tf in active_tfs:
                        try:
                            scan_smt_for_set(strategy, tf, active_tfs)
                        except Exception as e:
                            logger.error(f"Scan error {strategy} {tf}: {e}")
                
                logger.info(f"✅ Scan complete")
                time.sleep(SCAN_INTERVAL)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(60)
    
    def handle_commands(self):
        logger.info("🎮 Command handler started")
        
        while self.running:
            try:
                updates = get_telegram_updates(self.last_update_id)
                
                for update in updates:
                    self.last_update_id = update['update_id'] + 1
                    
                    if 'message' not in update:
                        continue
                    
                    message = update['message']
                    text = message.get('text', '').strip()
                    user_id = str(message['from']['id'])
                    
                    if text in ["/start", "/menu"]:
                        send_main_menu()
                    
                    elif text == "⚡ 5m Only":
                        update_user_preferences(user_id, timeframes=["5m"])
                        send_telegram("✅ Monitoring **5m only**")
                    
                    elif text == "🎯 15m Only":
                        update_user_preferences(user_id, timeframes=["15m"])
                        send_telegram("✅ Monitoring **15m only**")
                    
                    elif text == "📈 1h Only":
                        update_user_preferences(user_id, timeframes=["1h"])
                        send_telegram("✅ Monitoring **1h only**")
                    
                    elif text == "🌟 All Timeframes":
                        update_user_preferences(user_id, timeframes=["5m", "15m", "1h"])
                        send_telegram("✅ Monitoring **all timeframes**")
                    
                    elif text == "⏸️ Pause":
                        update_user_preferences(user_id, notifications=False)
                        send_telegram("⏸️ Signals **paused**")
                    
                    elif text == "📊 Statistics":
                        stats = get_statistics()
                        send_telegram(stats)
                    
                    elif text == "⚙️ Settings":
                        active_tfs, notif = get_user_preferences(user_id)
                        status = "✅ Active" if notif else "⏸️ Paused"
                        msg = f"⚙️ **SETTINGS**\n\nStatus: {status}\nTFs: {', '.join(active_tfs)}"
                        send_telegram(msg)
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Command error: {e}")
                time.sleep(5)

# ==========================================
# 🚀 MAIN
# ==========================================
if __name__ == "__main__":
    try:
        init_database()
        bot = TradingBotMonitor()
        bot.start()
    except Exception as e:
        logger.critical(f"CRITICAL: {e}")
        traceback.print_exc()
        sys.exit(1)
