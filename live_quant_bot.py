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
    return [row.find_all(['td', 'th'])[0].text.strip().replace('.', '-') for row in table.find_all('tr')[1:] if row.find_all(['td', 'th'])]

def get_nasdaq100_tickers():
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table', {'class': 'wikitable'})
    if not tables:
        raise ValueError("Could not find Nasdaq-100 table.")
    return [row.find_all(['td', 'th'])[0].text.strip().replace('.', '-') for row in tables[0].find_all('tr')[1:] if row.find_all(['td', 'th']) and len(row.find_all(['td', 'th'])[0].text.strip()) <= 6]

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
# 3. QUANT & SENTIMENT ENGINE (CLEAN YFINANCE)
# ==========================================
def fetch_live_bars(ticker):
    try:
        df = yf.download(ticker, period="10d", interval="1h", progress=False)
        if df is None or df.empty or len(df) < 20:
            return None
        
        # Flatten multi-index columns if present in newer yfinance releases
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.columns = [c.lower() for c in df.columns]
        
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            return None
            
        return df
    except Exception as e:
        print(f"⚠️ Yahoo Finance Error fetching bars for {ticker}: {e}")
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
        print(f"⚠️ Finnhub Sentiment API Error for {ticker}: {e}")
        return 0.50

def run_quant_analysis(ticker):
    bars = fetch_live_bars(ticker)
    if bars is None:
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

    if rsi < 35 and current_price > sma_20 and sma_20 > sma_50 and has_volume_support:
        nlp_score = analyze_news_sentiment(ticker)
        tech_score = (100 - rsi) / 100.0
        conviction = int(((tech_score * 0.5) + (nlp_score * 0.5)) * 100)
        
        if conviction >= 78:
            atr = float(ta.volatility.AverageTrueRange(high=bars['high'], low=bars['low'], close=close, window=14).average_true_range().iloc[-1])
            stop_loss = round(current_price - (1.5 * atr), 2)
            take_profit = round(current_price + (3.0 * atr), 2)
            
            return {
                "ticker": ticker, "price": current_price, "rsi": round(rsi, 1),
                "nlp": round(nlp_score, 2), "conviction": conviction,
                "stop_loss": stop_loss, "take_profit": take_profit
            }
    return None

# ==========================================
# MAIN RUNNER (HIGH SPEED, REAL-TIME ALERTS)
# ==========================================
def main():
    print("🚀 Initializing High-Speed Quant Scan (S&P 500 & Nasdaq-100)...")
    
    sp500_list = get_sp500_tickers()
    nasdaq_list = get_nasdaq100_tickers()
    
    universe = list(set(sp500_list) | set(nasdaq_list))
    print(f"Tracking {len(universe)} tickers. Running full-speed scan...")
    
    signals_found = 0
    for ticker in universe:
        if is_recent_signal(ticker, hours=24):
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
            # Real-time dispatch and immediate logging
            send_telegram_alert(msg)
            log_signal_to_csv(result)
            signals_found += 1
        
    print(f"Scan complete. Found {signals_found} new actionable signals.")

if __name__ == "__main__":
    main()
