import os
import csv
from pathlib import Path
import requests
import pandas as pd
import yfinance as yf
import ta
import finnhub
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==========================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Initialize Sentiment Client
vader_analyzer = SentimentIntensityAnalyzer()
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY) if FINNHUB_API_KEY else None

# ==========================================
# 1. UNIVERSE FETCHERS (S&P 500 & NASDAQ-100)
# ==========================================
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    if not table:
        raise ValueError("Could not find S&P 500 table.")
    
    tickers = []
    for row in table.find_all('tr')[1:]:
        cells = row.find_all(['td', 'th'])
        if cells:
            ticker = cells[0].text.strip().replace('.', '-')
            if ticker:
                tickers.append(ticker)
    return tickers

def get_nasdaq100_tickers():
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table', {'class': 'wikitable'})
    if not tables:
        raise ValueError("Could not find Nasdaq-100 table.")
        
    tickers = []
    for row in tables[0].find_all('tr')[1:]:
        cells = row.find_all(['td', 'th'])
        if cells:
            ticker = cells[0].text.strip().replace('.', '-')
            if ticker and len(ticker) <= 6:
                tickers.append(ticker)
    return tickers

# ==========================================
# 2. TELEGRAM & LOGGING SYSTEMS
# ==========================================
def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=5)
    except Exception as e:
        print(f"⚠️ Telegram Alert Error: {e}")

def is_recent_signal(ticker, hours=24):
    file_path = "backtest_signals.csv"
    if not Path(file_path).is_file():
        return False
    try:
        df = pd.read_csv(file_path)
        if df.empty or "Ticker" not in df.columns or "Timestamp" not in df.columns:
            return False
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent = df[(df["Ticker"] == ticker) & (df["Timestamp"] >= cutoff_time)]
        return not recent.empty
    except Exception as e:
        print(f"⚠️ Error reading backtest log: {e}")
        return False

def log_signal_to_csv(result):
    file_path = "backtest_signals.csv"
    file_exists = Path(file_path).is_file()
    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Ticker", "Exchange", "Price", "RSI", "NLP_Score", "Conviction", "Stop_Loss", "Take_Profit"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            result["ticker"], 
            result["exchange"],
            result["price"], 
            result["rsi"], 
            result["nlp"], 
            result["conviction"], 
            result["stop_loss"], 
            result["take_profit"]
        ])

# ==========================================
# 3. QUANT & SENTIMENT ENGINE (DETAILED LOGGING)
# ==========================================
def fetch_live_bars(ticker):
    try:
        df = yf.download(ticker, period="10d", interval="1h", progress=False)
        if df is None or df.empty or len(df) < 20:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.columns = [c.lower() for c in df.columns]
        
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            return None
            
        return df
    except Exception:
        return None

def analyze_news_sentiment(ticker):
    if not finnhub_client:
        return 0.50
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        news = finnhub_client.company_news(ticker, _from=today, to=today)
        if not news:
            return 0.50
        
        compound_scores = []
        for item in news[:5]:
            text = f"{item.get('headline', '')} {item.get('summary', '')}"
            if text.strip():
                scores = vader_analyzer.polarity_scores(text)
                compound_scores.append(scores['compound'])
        
        if not compound_scores:
            return 0.50
            
        avg_compound = sum(compound_scores) / len(compound_scores)
        normalized_score = 0.50 + (max(0.0, avg_compound) * 0.45)
        return round(max(0.50, min(0.95, normalized_score)), 2)
    except Exception as e:
        return 0.50

def run_quant_analysis(ticker):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bars = fetch_live_bars(ticker)
    if bars is None:
        print(f"[{now_str}] ⚪ {ticker} | Insufficient or invalid bar data -> NO BUY", flush=True)
        return None
        
    close = bars['close']
    volume = bars['volume']
    current_price = float(close.iloc[-1])
    
    rsi = float(ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1])
    sma_20 = float(ta.trend.SMAIndicator(close=close, window=20).sma_indicator().iloc[-1])
    sma_50 = float(ta.trend.SMAIndicator(close=close, window=50).sma_indicator().iloc[-1])
    
    avg_volume = float(volume.rolling(20).mean().iloc[-1])
    current_volume = float(volume.iloc[-1])
    has_volume_support = current_volume >= (avg_volume * 1.2)

    rsi_pass = rsi < 35
    trend_pass = (current_price > sma_20) and (sma_20 > sma_50)
    
    metrics_log = (
        f"[{now_str}] 🔍 {ticker} "
        f"| Price: ${current_price:.2f} "
        f"| RSI: {rsi:.1f} (Req < 35: {rsi_pass}) "
        f"| SMA20: ${sma_20:.2f} "
        f"| SMA50: ${sma_50:.2f} "
        f"| Trend OK: {trend_pass} "
        f"| Vol Support: {has_volume_support}"
    )

    if not rsi_pass:
        print(f"{metrics_log} -> NO BUY (RSI too high)", flush=True)
        return None
    if not trend_pass:
        print(f"{metrics_log} -> NO BUY (Trend criteria failed)", flush=True)
        return None
    if not has_volume_support:
        print(f"{metrics_log} -> NO BUY (Insufficient volume support)", flush=True)
        return None

    nlp_score = analyze_news_sentiment(ticker)
    tech_score = (100 - rsi) / 100.0
    conviction = int(((tech_score * 0.5) + (nlp_score * 0.5)) * 100)
    
    full_log = f"{metrics_log} | NLP Sentiment: {nlp_score} | Conviction: {conviction}% (Req >= 78%)"

    if conviction >= 78:
        atr = float(ta.volatility.AverageTrueRange(high=bars['high'], low=bars['low'], close=close, window=14).average_true_range().iloc[-1])
        stop_loss = round(current_price - (1.5 * atr), 2)
        take_profit = round(current_price + (3.0 * atr), 2)
        
        print(f"{full_log} -> 🟢 **BUY SIGNAL TRIGGERED!**", flush=True)
        return {
            "ticker": ticker, "price": current_price, "rsi": round(rsi, 1),
            "nlp": round(nlp_score, 2), "conviction": conviction,
            "stop_loss": stop_loss, "take_profit": take_profit
        }
    else:
        print(f"{full_log} -> NO BUY (Conviction too low)", flush=True)
        return None

# ==========================================
# MAIN RUNNER
# ==========================================
def main():
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{start_time}] 🚀 Initializing High-Speed Quant Scan (S&P 500 & Nasdaq-100)...", flush=True)
    
    sp500_list = get_sp500_tickers()
    nasdaq_list = get_nasdaq100_tickers()
    
    universe = sorted(list(set(sp500_list) | set(nasdaq_list)))
    total_tickers = len(universe)
    print(f"[{start_time}] 📊 Tracking {total_tickers} unique tickers. Starting scan...", flush=True)
    
    signals_found = 0
    for idx, ticker in enumerate(universe, 1):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if is_recent_signal(ticker, hours=24):
            print(f"[{now_str}] ⏳ [{idx}/{total_tickers}] {ticker} skipped (24h cooldown active)", flush=True)
            continue
            
        result = run_quant_analysis(ticker)
        if result:
            if ticker in set(nasdaq_list):
                result["exchange"] = "NASDAQ-100"
            else:
                result["exchange"] = "S&P 500"
                
            msg = (
                f"🚨 *MANUAL TRADE SETUP* 🚨\n\n"
                f"*Ticker:* `{result['ticker']}`\n"
                f"*Exchange:* `{result['exchange']}`\n"
                f"*Entry Price:* `${result['price']:.2f}`\n"
                f"*Stop Loss:* `${result['stop_loss']:.2f}`\n"
                f"*Take Profit:* `${result['take_profit']:.2f}`\n"
                f"*Conviction:* `{result['conviction']}%`"
            )
            send_telegram_alert(msg)
            log_signal_to_csv(result)
            signals_found += 1
        
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{end_time}] ✅ Scan complete. Found {signals_found} new actionable signals.", flush=True)

if __name__ == "__main__":
    main()
