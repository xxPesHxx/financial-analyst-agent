import random
from app.guardrails import validate_ticker

def get_stock_price(ticker: str) -> dict:
    validate_ticker(ticker)
    # Mock price
    price = round(random.uniform(100, 200), 2)
    return {"ticker": ticker, "price": price, "currency": "USD"}

def technical_analysis_forecast(ticker: str, horizon: str) -> dict:
    validate_ticker(ticker)
    # Mock NN: Ensure strict I/O
    # logic: if ticker starts with 'A', it's UPTREND, else DOWNTREND (deterministic for testing maybe? or random)
    # Prompt says "Simulates your Neural Network prediction (you can mock the logic, but the I/O must be strict)"
    signal = "UPTREND" if random.random() > 0.5 else "DOWNTREND"
    return {"ticker": ticker, "horizon": horizon, "forecast": signal, "confidence": round(random.uniform(0.5, 0.99), 2)}

def get_rsi(ticker: str) -> dict:
    validate_ticker(ticker)
    rsi = round(random.uniform(30, 70), 2)
    return {"ticker": ticker, "rsi": rsi}
