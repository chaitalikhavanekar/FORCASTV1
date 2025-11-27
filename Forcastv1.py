import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="AuroraQuant – ARIMA Stock Forecasting App",
    layout="wide",
)

# ================== SIMPLE UI HEADER ==================
st.title("📈 AuroraQuant – ARIMA Stock Forecasting App")
st.write("Advanced ARIMA Forecasting + Indicators + Macro ARIMAX comparison.")


# ================== HELPER FUNCTIONS ==================

def fetch_price_data(ticker, start, end, freq="Monthly"):
    """Download historical stock prices from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("Ticker data not found. Check symbol or timeframe.")
    close = df["Close"].dropna()
    if freq == "Monthly":
        close = close.resample("M").last()
    return close


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def compute_volatility(series, freq="Monthly", window=6):
    ret = series.pct_change()
    if freq == "Monthly":
        vol = ret.rolling(window).std() * np.sqrt(12)
    else:
        vol = ret.rolling(window).std() * np.sqrt(252)
    return vol


def metrics(y_true, y_pred):
    """Custom MAE, RMSE, MAPE (no sklearn)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    return mae, rmse, mape


def auto_arima_search(series):
    """Mini auto-ARIMA grid search."""
    best_aic = np.inf
    best_order = (1, 1, 1)

    for p in range(4):
        for d in range(3):
            for q in range(4):
                if p == d == q == 0:
                    continue
                try:
                    model = ARIMA(series, order=(p,d,q))
                    res = model.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p,d,q)
                except:
                    continue
    return best_order, best_aic


def fetch_macro(start, end):
    tickers = {
        "NIFTY50": "^NSEI",
        "CrudeOil": "CL=F",
        "USDINR": "INR=X"
    }
    out = {}
    for name, t in tickers.items():
        df = yf.download(t, start=start, end=end, progress=False)
        if not df.empty:
            out[name] = df["Close"].dropna()
    return out


def align_macro(price, macro_dict):
    df = pd.DataFrame({"Stock": price})
    for name, s in macro_dict.items():
        df[name] = s
    df = df.dropna()
    y = df["Stock"]
    X = df.drop(columns=["Stock"])
    return y, X


# ================== SIDEBAR ==================
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Stock ticker (NSE example: RELIANCE.NS)", value="RELIANCE.NS")

time_period = st.sidebar.selectbox(
    "Select Timeframe",
    ["2010–2018", "2021–2025", "Max Available"]
)

freq = st.sidebar.radio("Data Frequency", ["Monthly", "Daily"])

auto_arima = st.sidebar.checkbox("Auto ARIMA (recommended)", value=True)

if not auto_arima:
    p = st.sidebar.slider("p", 0, 4, 1)
    d = st.sidebar.slider("d", 0, 2, 1)
    q = st.sidebar.slider("q", 0, 4, 1)
    manual_order = (p, d, q)
else:
    manual_order = None

use_arimax = st.sidebar.checkbox("Enable ARIMAX (Macro Exogenous)", value=False)

forecast_horizon = st.sidebar.slider("Forecast Months", 6, 24, 12)

run = st.sidebar.button("🚀 Run Forecast")


# ================== MAIN APP ==================

if not run:
    st.info("Configure the options in the left sidebar and click **Run Forecast**.")
else:
    try:
        # ------------------------ Timeframe mapping ------------------------
        if time_period == "2010–2018":
            start = "2010-01-01"
            end = "2019-01-01"
            test_year = 2018
        elif time_period == "2021–2025":
            start = "2021-01-01"
            end = "2026-01-01"
            test_year = 2025
        else:
            start = "2000-01-01"
            end = datetime.today().strftime("%Y-%m-%d")
            test_year = datetime.today().year - 1

        # ------------------------ Fetch Data ------------------------
        price = fetch_price_data(ticker, start, end, freq=freq)

        st.subheader(f"📊 Historical Price – {ticker}")
        st.line_chart(price)

        # Indicators
        rsi = compute_rsi(price)
        macd, macd_signal, macd_hist = compute_macd(price)
        vol = compute_volatility(price, freq=freq)

        st.subheader("📌 Indicators")
        c1, c2 = st.columns(2)

        with c1:
            st.write("RSI")
            st.line_chart(rsi)
        with c2:
            st.write("MACD")
            st.line_chart(pd.DataFrame({"MACD": macd, "Signal": macd_signal}))

        st.write("Volatility")
        st.line_chart(vol)

        # ------------------------ Train/Test Split ------------------------
        train = price[price.index < f"{test_year}-01-01"]
        test = price[price.index >= f"{test_year}-01-01"]

        # ------------------------ ARIMA Order ------------------------
        if auto_arima:
            order, aic = auto_arima_search(train)
        else:
            order = manual_order

        st.subheader(f"🔢 ARIMA Order Selected: {order}")

        # ------------------------ Train ARIMA ------------------------
        model_train = ARIMA(train, order=order)
        res_train = model_train.fit()

        fc_test_res = res_train.get_forecast(steps=len(test))
        fc_test = fc_test_res.predicted_mean

        # ------------------------ Train Full Model ------------------------
        model_full = ARIMA(price, order=order)
        res_full = model_full.fit()

        fc_future_res = res_full.get_forecast(steps=forecast_horizon)
        fc_future = fc_future_res.predicted_mean

        # Make future index
        last_date = price.index[-1]
        if freq == "Monthly":
            future_index = pd.date_range(last_date + pd.offsets.MonthEnd(1), periods=forecast_horizon, freq="M")
        else:
            future_index = pd.date_range(last_date + pd.offsets.Day(1), periods=forecast_horizon, freq="D")

        fc_future.index = future_index

        # ------------------------ Display ARIMA Forecast ------------------------
        st.subheader("🔮 ARIMA Forecast vs Actual Test Year")
        df_test_plot = pd.DataFrame({
            "Train": train,
            "Actual (Test)": test,
            "Forecast (Test)": fc_test
        })
        st.line_chart(df_test_plot)

        # ------------------------ Metrics ------------------------
        if len(test) > 0:
            mae, rmse, mape = metrics(test, fc_test)
            st.write("### 📏 Forecast Error Metrics (Test Year)")
            st.metric("MAE", f"{mae:.2f}")
            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("MAPE (%)", f"{mape:.2f}")

        # ------------------------ Future Path ------------------------
        st.subheader("🚀 Future Forecast")
        df_future_plot = pd.concat([price.rename("Historical"), fc_future.rename("Forecast")], axis=0)
        st.line_chart(df_future_plot)

        # ------------------------ Macro ARIMAX ------------------------
        if use_arimax:
            st.subheader("🌐 ARIMAX (Macro Exogenous Variables)")
            macro_data = fetch_macro(start, end)

            if macro_data:
                y, X = align_macro(price, macro_data)
                y_train = y[y.index < f"{test_year}-01-01"]
                y_test = y[y.index >= f"{test_year}-01-01"]
                X_train = X.loc[y_train.index]
                X_test = X.loc[y_test.index]

                if len(y_train) > 10:
                    arimax = SARIMAX(y_train, exog=X_train, order=order)
                    arimax_res = arimax.fit()

                    arimax_fc = arimax_res.predict(
                        start=y_test.index[0],
                        end=y_test.index[-1],
                        exog=X_test
                    )

                    df_compare = pd.DataFrame({
                        "Actual": y_test,
                        "ARIMA": fc_test.loc[y_test.index],
                        "ARIMAX": arimax_fc
                    })

                    st.line_chart(df_compare)

                else:
                    st.warning("Not enough aligned macro data for ARIMAX.")

            else:
                st.warning("Macro data unavailable.")

    except Exception as e:
        st.error(f"Error: {e}")
