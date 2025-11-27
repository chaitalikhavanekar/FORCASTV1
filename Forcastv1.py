# ------------------- Forcastv1.py (FULL CLEAN FILE) -------------------
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="AuroraQuant – ARIMA Stock Forecasting",
    layout="wide",
)

st.title("📈 AuroraQuant – ARIMA Stock Forecasting")
st.write("ARIMA-based price forecasting with basic indicators and error metrics.")


# ========== HELPER FUNCTIONS ==========

def fetch_price_data(ticker: str, start: str, end: str, freq: str = "Monthly") -> pd.Series:
    """Download historical stock prices and return Close series."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No data downloaded – check ticker symbol or timeframe.")
    close = df["Close"].dropna()
    if freq == "Monthly":
        close = close.resample("M").last()
    return close


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    """MACD indicator."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def compute_volatility(series: pd.Series, freq: str = "Monthly", window: int = 6) -> pd.Series:
    """Rolling annualised volatility."""
    returns = series.pct_change()
    if freq == "Monthly":
        vol = returns.rolling(window).std() * np.sqrt(12)
    else:
        vol = returns.rolling(window).std() * np.sqrt(252)
    return vol


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


def auto_arima_order(series: pd.Series):
    """Tiny grid search for best (p,d,q) by AIC."""
    best_aic = np.inf
    best_order = (1, 1, 1)
    for p in range(0, 4):
        for d in range(0, 3):
            for q in range(0, 4):
                if p == d == q == 0:
                    continue
                try:
                    model = ARIMA(series, order=(p, d, q))
                    res = model.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best_order, best_aic


# ========== SIDEBAR CONTROLS ==========

st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Stock ticker (e.g., RELIANCE.NS)", value="RELIANCE.NS")

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["2010–2018", "2021–2025", "Max Available"]
)

freq = st.sidebar.radio("Data Frequency", ["Monthly", "Daily"])

auto_order = st.sidebar.checkbox("Auto ARIMA order (AIC based)", value=True)
if not auto_order:
    p = st.sidebar.slider("p (AR)", 0, 4, 1)
    d = st.sidebar.slider("d (diff)", 0, 2, 1)
    q = st.sidebar.slider("q (MA)", 0, 4, 1)
    manual_order = (p, d, q)
else:
    manual_order = None

forecast_horizon = st.sidebar.slider("Forecast horizon (months)", 6, 24, 12)

run_btn = st.sidebar.button("🚀 Run ARIMA Forecast")


# ========== MAIN LOGIC ==========

if not run_btn:
    st.info("Set your options in the sidebar and click **Run ARIMA Forecast**.")
else:
    try:
        # ----- Map timeframe -----
        if timeframe == "2010–2018":
            start = "2010-01-01"
            end = "2019-01-01"
            test_year = 2018
        elif timeframe == "2021–2025":
            start = "2021-01-01"
            end = "2026-01-01"
            test_year = 2025
        else:  # Max available
            start = "2000-01-01"
            end = datetime.today().strftime("%Y-%m-%d")
            test_year = datetime.today().year - 1

        # ----- Fetch data -----
        price = fetch_price_data(ticker, start, end, freq=freq)

        st.subheader(f"Price History – {ticker}")
        st.line_chart(price.rename("Close"))

        # ----- Indicators -----
        rsi = compute_rsi(price)
        macd, macd_signal, macd_hist = compute_macd(price)
        vol = compute_volatility(price, freq=freq)

        st.subheader("Indicators")

        col1, col2 = st.columns(2)
        with col1:
            st.write("RSI")
            st.line_chart(rsi.rename("RSI"))

        with col2:
            st.write("MACD")
            macd_df = pd.DataFrame({
                "MACD": macd,
                "Signal": macd_signal
            })
            st.line_chart(macd_df)

        st.write("Volatility (Annualised)")
        st.line_chart(vol.rename("Volatility"))

        # ----- Train/Test split -----
        train = price[price.index < f"{test_year}-01-01"]
        test = price[price.index >= f"{test_year}-01-01"]

        if len(train) < 20:
            st.warning("Not enough data for ARIMA in this timeframe. Try 'Max Available'.")
        else:
            # ----- Choose ARIMA order -----
            if auto_order:
                order, best_aic = auto_arima_order(train)
                st.subheader(f"Selected ARIMA Order: {order} (best AIC = {best_aic:.2f})")
            else:
                order = manual_order
                st.subheader(f"Using manual ARIMA Order: {order}")

            # ----- Fit ARIMA on train + forecast test -----
            model_train = ARIMA(train, order=order)
            res_train = model_train.fit()
            fc_test_res = res_train.get_forecast(steps=len(test))
            fc_test = fc_test_res.predicted_mean

            # ----- Metrics -----
            if len(test) > 0:
                mae, rmse, mape = metrics(test, fc_test)
            else:
                mae = rmse = mape = np.nan

            # ----- Fit ARIMA on full data + forecast future -----
            model_full = ARIMA(price, order=order)
            res_full = model_full.fit()
            fc_future_res = res_full.get_forecast(steps=forecast_horizon)
            fc_future = fc_future_res.predicted_mean

            last_date = price.index[-1]
            if freq == "Monthly":
                future_index = pd.date_range(last_date + pd.offsets.MonthEnd(1),
                                             periods=forecast_horizon, freq="M")
            else:
                future_index = pd.date_range(last_date + pd.offsets.Day(1),
                                             periods=forecast_horizon, freq="D")
            fc_future.index = future_index

            # ----- Plot train/test/forecast -----
            st.subheader("ARIMA – Train vs Test Forecast")
            df_test_plot = pd.DataFrame({
                "Train": train,
                "Actual (Test)": test,
                "Forecast (Test)": fc_test
            })
            st.line_chart(df_test_plot)

            st.subheader("Forecast Error Metrics (Test Year)")
            cmae, crmse, cmape = st.columns(3)
            with cmae:
                st.metric("MAE", f"{mae:.2f}")
            with crmse:
                st.metric("RMSE", f"{rmse:.2f}")
            with cmape:
                st.metric("MAPE (%)", "N/A" if np.isnan(mape) else f"{mape:.2f}")

            # ----- Future path -----
            st.subheader("Future Forecast Path")
            df_future_plot = pd.concat(
                [price.rename("Historical"), fc_future.rename("Forecast")],
                axis=0
            )
            st.line_chart(df_future_plot)

            # ----- Downloads -----
            hist_df = price.to_frame(name="Close")
            csv_hist = hist_df.to_csv().encode("utf-8")
            st.download_button(
                "⬇️ Download historical data (CSV)",
                csv_hist,
                file_name=f"{ticker}_historical.csv",
                mime="text/csv"
            )

            fc_df = fc_future.to_frame(name="Forecast")
            csv_fc = fc_df.to_csv().encode("utf-8")
            st.download_button(
                "⬇️ Download forecast data (CSV)",
                csv_fc,
                file_name=f"{ticker}_forecast.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Check ticker, timeframe, or try switching Monthly/Daily.")
# ------------------- END FILE -------------------
