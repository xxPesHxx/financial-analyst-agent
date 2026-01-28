import random
from app.guardrails import validate_ticker

def get_stock_price(ticker: str) -> dict:
    validate_ticker(ticker)
    price = round(random.uniform(100, 200), 2)
    return {"ticker": ticker, "price": price, "currency": "USD"}

def technical_analysis_forecast(ticker: str, horizon: str) -> dict:
    validate_ticker(ticker)
    signal = "UPTREND" if random.random() > 0.5 else "DOWNTREND"
    return {"ticker": ticker, "horizon": horizon, "forecast": signal, "confidence": round(random.uniform(0.5, 0.99), 2)}

def get_rsi(ticker: str) -> dict:
    validate_ticker(ticker)
    rsi = round(random.uniform(30, 70), 2)
    return {"ticker": ticker, "rsi": rsi}
