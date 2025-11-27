import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ================== PAGE CONFIG & CSS ==================

st.set_page_config(
    page_title="AuroraQuant – ARIMA Stock Lab",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #eef2ff 0, #ffffff 50%, #f9fafb 100%);
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        margin-bottom: 0.1rem;
        color: #111827;
    }
    .sub-title {
        text-align: center;
        font-size: 0.95rem;
        font-style: italic;
        color: #6b7280;
        margin-bottom: 1.4rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111827;
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
    .section-caption {
        font-size: 0.8rem;
        color: #6b7280;
        margin-bottom: 0.6rem;
    }
    .info-card {
        padding: 16px 18px;
        border-radius: 16px;
        background: rgba(255,255,255,0.95);
        border: 1px solid rgba(148,163,184,0.35);
        box-shadow: 0 18px 40px rgba(15,23,42,0.07);
    }
    .info-label {
        font-size: 0.85rem;
        color: #4b5563;
        font-weight: 600;
        margin-bottom: 0.1rem;
    }
    .info-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #111827;
    }
    .info-sub {
        font-size: 0.78rem;
        color: #6b7280;
    }
    .divider-soft {
        border-bottom: 1px solid rgba(148,163,184,0.45);
        margin: 1.0rem 0 1.0rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================== HELPER FUNCTIONS ==================

def fetch_price_data(ticker: str, start: str, end: str, freq: str = "Monthly") -> pd.Series:
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise ValueError("No data downloaded – check ticker or timeframe.")
    close = data["Close"].dropna()
    if freq == "Monthly":
        close = close.resample("M").last()
    return close


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def compute_volatility(series: pd.Series, freq: str = "Monthly", window: int = 6):
    returns = series.pct_change()
    if freq == "Monthly":
        vol = returns.rolling(window).std() * np.sqrt(12)
    else:
        vol = returns.rolling(window).std() * np.sqrt(252)
    return vol, returns


def metrics(y_true: pd.Series, y_pred: pd.Series):
    """Custom MAE, RMSE, MAPE (no sklearn)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    return mae, rmse, mape


def auto_arima_order(series: pd.Series, max_p=2, max_d=2, max_q=2):
    """Very small grid search on p,d,q using AIC."""
    best_aic = np.inf
    best_order = (1, 1, 1)
    for p in range(0, max_p + 1):
        for d in range(0, max_d + 1):
            for q in range(0, max_q + 1):
                if p == d == q == 0:
                    continue
