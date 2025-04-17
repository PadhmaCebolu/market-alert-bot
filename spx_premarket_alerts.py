import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import csv
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# =============================
# 📋 Utility Functions
# =============================

def classify_headlines_openai_bulk(headlines):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're a financial sentiment classifier. For each headline, respond with just 📈, 📉, or 🔹. Give one per line in the same order."},
                {"role": "user", "content": "\n".join(headlines)}
            ],
            max_tokens=50
        )
        result_text = response.choices[0].message.content.strip()
        sentiments = result_text.splitlines()
        return sentiments
    except Exception as e:
        print("❌ OpenAI classification failed:", e)
        return ["🔹"] * len(headlines)


# =============================
# 📋 News Scrapers
# =============================

def scrape_headlines(url, selector, base_url=""):
    headlines = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
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
    headlines_raw = (
        scrape_headlines("https://macenews.com/", ".elementor-heading-title") +
        scrape_headlines("https://www.cnbc.com/world/?region=world", "a.Card-title") +
        scrape_headlines("https://www.reuters.com/", "a[data-testid='Heading']", base_url="https://www.reuters.com")
    )

    classified = classify_headlines_openai_bulk(headlines_raw)

    enhanced_news = []
    for original, sentiment in zip(headlines_raw, classified):
        score = {"📈": 3, "📉": -3, "🔹": 0}.get(sentiment, 0)
        enhanced_news.append((sentiment, score, f"{sentiment} {original}"))

    return enhanced_news


# =============================
# 📋 Market Data
# =============================

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

def get_spx():
    return get_price_from_investing("https://www.investing.com/indices/us-spx-500")

def get_vix():
    return get_price_from_investing("https://www.investing.com/indices/volatility-s-p-500")

def get_es():
    return get_price_from_investing("https://www.investing.com/indices/us-spx-500-futures")

# =============================
# 📋 Analysis & Bias
# =============================

def estimate_direction(spx, es, sentiment_score, vix):
    score = 0
    reasons = []
    gap = es - spx if isinstance(spx, float) and isinstance(es, float) else 0

    if gap > 10:
        score += 1
        reasons.append("ES futures lead SPX → bullish")
    elif gap < -10:
        score -= 1
        reasons.append("ES futures lag SPX → bearish")

    if sentiment_score >= 3:
        score += 1
        reasons.append("Positive news bias")
    elif sentiment_score <= -3:
        score -= 1
        reasons.append("Negative news bias")

    if isinstance(vix, float) and vix > 30:
        score -= 1
        reasons.append("High VIX (>30) → bearish weight")

    if score >= 1:
        return "📈 Bullish", reasons
    elif score <= -1:
        return "📉 Bearish", reasons
    else:
        return "⚖️ Neutral", reasons

def calculate_vix_move(spx, vix, bias):
    try:
        if isinstance(spx, float) and isinstance(vix, float):
            move = (spx * vix / 100) / (252 ** 0.5)
            move_points = round(move, 2)
            return (-move_points if bias == "📉 Bearish" else move_points), f"{move_points} pts {'drop' if bias == '📉 Bearish' else 'rise'} expected"
    except Exception as e:
        print(f"⚠️ Error calculating VIX move: {e}")
    return "N/A", "N/A"

# =============================
# 📅 Logging
# =============================

def log_premarket_prediction(date, spx, es, vix, sentiment_score, direction, move_pts):
    log_file = "market_predictions.csv"
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["date", "spx", "es", "vix", "sentiment_score", "predicted_trend", "predicted_move_pts"])
        writer.writerow([date, spx, es, vix, sentiment_score, direction, move_pts])

# =============================
# 📧 Email Notifier
# =============================

def send_email(subject, body, to_email):
    try:
        message = MIMEMultipart()
        message["From"] = os.getenv("EMAIL_USER")
        message["To"] = to_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
            server.send_message(message)
            print("✅ Email sent.")
    except Exception as e:
        print("❌ Email failed:", e)

# =============================
# 📊 Main
# =============================

def main():
    today = datetime.date.today()
    spx, vix, es = get_spx(), get_vix(), get_es()
    news = get_all_market_news()
    sentiment_score = sum(score for _, score, _ in news)
    direction, reasons = estimate_direction(spx, es, sentiment_score, vix)
    move_pts, move_msg = calculate_vix_move(spx, vix, direction)

    alert = [
        f"📊 Pre-Market Alert for {today}",
        f"🔹 SPX: {spx}  🔺 VIX: {vix}  📉 ES: {es}",
        f"\n📰 Headlines:", *[f"- {h}" for _, _, h in news],
        f"\n📊 Market Bias: {direction}", *[f"- {r}" for r in reasons],
        f"\n📉 VIX-Derived Expected Move: {move_msg}"
    ]

    full_message = "\n".join(alert)
    print(full_message)

    log_premarket_prediction(today, spx, es, vix, sentiment_score, direction, move_pts)
    send_email("📊 Pre-Market Alert", full_message, os.getenv("EMAIL_TO"))

if __name__ == "__main__":
    main()
