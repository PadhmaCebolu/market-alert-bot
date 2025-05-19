import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import csv
import openai
from fredapi import Fred
import pytz
import re
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import yfinance as yf

# Load environment variables
load_dotenv()
EMAIL = os.getenv("MC_EMAIL")
PASSWORD = os.getenv("MC_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")

# Load API Keys
with open("API Keys/apikey_openai.txt", "r") as f:
    openai.api_key = f.readline().strip()
with open("API Keys/apikey_finnhub.txt", "r") as f:
    finnhub_api_key = f.readline().strip()
with open("API Keys/apikey_marketaux.txt", "r") as f:
    marketaux_api_key = f.readline().strip()
with open("API Keys/apikey_fred.txt", "r") as f:
    fred_api_key = f.readline().strip()
fred = Fred(api_key=fred_api_key)

DOWNLOAD_DIR = os.path.join(os.getcwd(), "data")
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)


def get_implied_move_yfinance(ticker="SPY", sentiment_score=None):
    try:
        stock = yf.Ticker(ticker)
        spot_price = stock.history(period="1d")["Close"].iloc[-1]
        spot_strike = round(spot_price / 5) * 5
        expirations = stock.options
        if not expirations:
            print("⚠️ No options data available.")
            return None
        expiration = expirations[0]
        opt_chain = stock.option_chain(expiration)
        calls = opt_chain.calls
        puts = opt_chain.puts
        call_row = calls.iloc[(calls['strike'] - spot_price).abs().argsort()[:1]]
        put_row = puts.iloc[(puts['strike'] - spot_price).abs().argsort()[:1]]
        call_ask = float(call_row["ask"].values[0])
        put_ask = float(put_row["ask"].values[0])
        straddle_cost = call_ask + put_ask
        implied_move_pct = (straddle_cost / spot_price) * 100

        if sentiment_score is not None:
            sign = '+' if sentiment_score >= 0 else '-'
            return f"{sign}{round(implied_move_pct, 2)}%"
        else:
            return f"±{round(implied_move_pct, 2)}%"
    except Exception as e:
        print("❌ Error fetching from yfinance:", e)
        return None


def classify_headlines_openai_weighted(headlines):
    prompt = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines)])
    system = "You're a financial analyst. For each headline, return a number between -5 and +5. Format: 1. -3"
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        output = response.choices[0].message.content.strip()
        print("🧠 GPT Output:\n", output)
        lines = output.splitlines()
        scores = []
        for line in lines:
            match = re.search(r"[-+]?[0-9]+", line)
            scores.append(int(match.group()) if match else 0)
        return scores
    except Exception as e:
        print("❌ Weighted classification failed:", e)
        return [0] * len(headlines)

def is_market_relevant(text):
    keywords = ["fed", "tariff", "rate", "inflation", "yields", "bond", "treasury", "earnings", "revenue",
                "stocks", "markets", "recession", "jobless", "cpi", "ppi", "gdp", "volatility"]
    return any(k in text.lower() for k in keywords)

def scrape_headlines(url, selector, base_url=""):
    headlines = []
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(res.text, 'html.parser')
        for el in soup.select(selector)[:10]:
            text = el.get_text(strip=True)
            link = el.get("href", "")
            if text and is_market_relevant(text):
                full_link = f"{base_url}{link}" if link.startswith("/") else link
                headlines.append(f"{text} - {full_link}")
    except Exception as e:
        print(f"⚠️ Error scraping {url}:", e)
    return headlines

def get_all_market_news():
    headlines_raw = scrape_headlines("https://www.cnbc.com/world/?region=world", "a.Card-title")
    try:
        res = requests.get(f"https://finnhub.io/api/v1/news?category=general&token={finnhub_api_key}").json()
        for item in res[:10]:
            if is_market_relevant(item.get("headline", "")):
                headlines_raw.append(f"{item['headline']} - {item['url']}")
    except Exception as e:
        print("❌ Finnhub news fetch failed:", e)
    try:
        res = requests.get(f"https://api.marketaux.com/v1/news/all?symbols=SPY&filter_entities=true&language=en&api_token={marketaux_api_key}").json()
        for article in res.get("data", [])[:10]:
            if is_market_relevant(article.get("title", "")):
                headlines_raw.append(f"{article['title']} - {article['url']}")
    except Exception as e:
        print("❌ Marketaux news fetch failed:", e)
    scores = classify_headlines_openai_weighted(headlines_raw)
    return list(zip(scores, headlines_raw))

def get_price_from_investing(url):
    try:
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(4)
        price = driver.find_element(By.CSS_SELECTOR, '[data-test="instrument-price-last"]').text.replace(",", "")
        driver.quit()
        return float(price)
    except Exception as e:
        print(f"⚠️ Error retrieving price from {url}:", e)
        return "N/A"

def get_spx(): return get_price_from_investing("https://www.investing.com/indices/us-spx-500")
def get_es(): return get_price_from_investing("https://www.investing.com/indices/us-spx-500-futures")
def get_vix(): return get_price_from_investing("https://www.investing.com/indices/volatility-s-p-500")

def get_previous_values():
    try:
        prev_vix = float(fred.get_series("VIXCLS").dropna().iloc[-2])
        prev_spx = float(fred.get_series("SP500").dropna().iloc[-2])
        return prev_spx, prev_vix
    except:
        return None, None

def get_weekly_trend_bias():
    try:
        spx_series = fred.get_series("SP500").dropna()
        last = spx_series.iloc[-1]
        week_ago = spx_series.iloc[-6]
        return 1 if last > week_ago else -1 if last < week_ago else 0
    except:
        return 0

def rule_based_market_bias(sentiment_score, vix, es, spx):
    bias = 0
    if sentiment_score > 0:
        bias += 1
    if es > spx:
        bias += 1
    if vix < 18:
        bias += 1
    return "📈 Bullish" if bias >= 2 else "📉 Bearish"

def log_market_features(spx, es, vix, prev_spx, prev_vix, implied_move, sentiment_score):
    vix_delta = (vix - prev_vix) / prev_vix if prev_vix and isinstance(vix, float) else 0
    futures_gap = es - prev_spx if prev_spx and isinstance(es, float) else 0
    trend_bias = get_weekly_trend_bias()
    row = {
        "date": datetime.date.today(),
        "weekly_trend": trend_bias,
        "sentiment_score": sentiment_score,
        "implied_move": implied_move,
        "vix": vix,
        "vix_delta": vix_delta,
        "futures_gap": futures_gap,
        "spx": spx
    }
    path = os.path.join(DOWNLOAD_DIR, "market_features.csv")
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def train_logistic_model():
    path = os.path.join(DOWNLOAD_DIR, "market_features.csv")
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df["target"] = (df["spx"].shift(-1) > df["spx"]).astype(int)
    df.dropna(inplace=True)

    features = ["weekly_trend", "sentiment_score", "implied_move", "vix", "vix_delta", "futures_gap"]
    X = df[features]
    y = df["target"]

    if len(set(y)) < 2:
        print("⚠️ Not enough class diversity in training labels. Skipping model training.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"🔍 Logistic Regression Accuracy: {accuracy:.2f}")
    return model

def predict_with_model(model, features_dict):
    if model is None:
        return "⚠️ No ML prediction due to training issue."
    import numpy as np
    features = np.array([[features_dict[k] for k in ["weekly_trend", "sentiment_score", "implied_move", "vix", "vix_delta", "futures_gap"]]])
    prob = model.predict_proba(features)[0][1]
    direction = "📈 Bullish" if prob > 0.5 else "📉 Bearish"
    print(f"🤖 ML Prediction: {direction} (Prob: {prob:.2f})")
    return direction


def send_email(subject, spx, vix, es, news, direction, reasons, move_msg, to_email):
    try:
        import pytz
        # Get current time in US/Eastern
        eastern = pytz.timezone('US/Eastern')
        current_time_est = datetime.datetime.now(eastern).strftime('%I:%M %p ET')

        message = MIMEMultipart("alternative")
        message["From"] = os.getenv("EMAIL_USER")
        message["To"] = to_email
        message["Subject"] = subject

        # Plaintext fallback
        body_text = f"""
📊 Pre-Market Alert for {datetime.date.today()}
🔹 SPX: {spx}  🔺 VIX: {vix}  📉 ES: {es}

📰 Headlines:
{chr(10).join([f"- {h}" for _, _, h in news])}

📊 Market Bias: {direction}
{chr(10).join([f"- {r}" for r in reasons])}

📉 Expected Move: {move_msg}
Generated by CDUS Trading Bot • {current_time_est}
        """

        # HTML version
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px; color: #333;">
            <h2 style="color: #0d6efd;">📊 Pre-Market Alert for {datetime.date.today()}</h2>
            <p>
                <strong>🔹 SPX:</strong> {spx} &nbsp;&nbsp;
                <strong>🔺 VIX:</strong> {vix} &nbsp;&nbsp;
                <strong>📉 ES:</strong> {es}
            </p>

            <h3>📰 Headlines:</h3>
            <ul>
                {''.join(f"<li>{h}</li>" for _, _, h in news)}
            </ul>

            <h3>📊 Market Bias: {direction}</h3>
                        
            <br>
            <p style="font-size: 0.9em; color: #888;">Generated by CDUS Trading Bot • {current_time_est}</p>
        </body>
        </html>
        """

        # Attach both parts
        message.attach(MIMEText(body_text, "plain"))
        message.attach(MIMEText(html, "html"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
            server.send_message(message)
            print("✅ Email sent.")
    except Exception as e:
        print("❌ Email failed:", e)

def main():
    today = datetime.date.today()
    news = get_all_market_news()
    sentiment_score = sum(score for score, _ in news)

    implied_move = get_implied_move_yfinance("SPY", sentiment_score)
    spx, es, vix = get_spx(), get_es(), get_vix()
    prev_spx, prev_vix = get_previous_values()

    log_market_features(spx, es, vix, prev_spx, prev_vix, implied_move.strip('±+-%'), sentiment_score)

    print("🧠 Classified Headlines with Sentiment:")
    for score, headline in news:
        print(f"{score:+d} {headline}")

    print(f"📊 Pre-Market Alert for {today}")
    print(f"SPX: {spx}, ES: {es}, VIX: {vix}")
    print(f"📈 Implied Move (SPY ATM): {implied_move}")
    print(f"Sentiment Score: {sentiment_score}")
    print("📉 Rule-based Bias:", rule_based_market_bias(sentiment_score, vix, es, spx))

    market_data_path = os.path.join(DOWNLOAD_DIR, "market_features.csv")
    if os.path.exists(market_data_path):
        df = pd.read_csv(market_data_path)
        if len(df) >= 10:
            model = train_logistic_model()
            vix_delta = (vix - prev_vix) / prev_vix if prev_vix else 0
            futures_gap = es - prev_spx if prev_spx else 0
            features_today = {
                "weekly_trend": get_weekly_trend_bias(),
                "sentiment_score": sentiment_score,
                "implied_move": float(implied_move.strip('±+-%')),
                "vix": vix,
                "vix_delta": vix_delta,
                "futures_gap": futures_gap
            }
            predict_with_model(model, features_today)
        else:
            print("⚠️ Not enough data to train ML model yet. Waiting for more logs...")

if __name__ == "__main__":
    main()