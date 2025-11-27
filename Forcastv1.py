import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# try to import auto_arima (advanced feature)
try:
    from pmdarima import auto_arima
    HAS_AUTO_ARIMA = True
except ImportError:
    HAS_AUTO_ARIMA = False

# ================== PAGE CONFIG & GLOBAL STYLE ==================

st.set_page_config(
    page_title="AuroraQuant – ARIMA Stock Lab",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #e0f2fe 0, #ffffff 36%, #eef2ff 100%);
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: .08em;
        margin-bottom: 0.0rem;
        text-transform: uppercase;
        color: #111827;
    }
    .sub-title {
        text-align: center;
        font-size: 0.95rem;
        font-style: italic;
        color: #6b7280;
        margin-bottom: 1.4rem;
    }
    .info-card {
        padding: 18px 20px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.93);
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.35);
    }
    .info-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #4b5563;
        margin-bottom: .35rem;
    }
    .metric-big {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #6b7280;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.2rem 0 0.4rem 0;
        color: #111827;
    }
    .section-caption {
        font-size: 0.82rem;
        color: #6b7280;
        margin-bottom: 0.35rem;
    }
    .divider-soft {
        border-bottom: 1px solid rgba(148, 163, 184, 0.4);
        margin: 1.0rem 0 1.0rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================== HELPER FUNCTIONS ==================

def fetch_price_data(ticker, start, end, freq="Monthly"):
    """Download OHLC data and return resampled close series."""
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise ValueError("No data found for this ticker / period.")
    close = data["Close"].dropna()
    if freq == "Monthly":
        close = close.resample("M").last()
    return close

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def compute_volatility(series, freq="Monthly", window=6):
    returns = series.pct_change()
    if freq == "Monthly":
        vol = returns.rolling(window=window).std() * np.sqrt(12)
    else:
        vol = returns.rolling(window=window).std() * np.sqrt(252)
    return vol, returns

def fit_fixed_arima(series, order):
    model = ARIMA(series, order=order)
    return model.fit()

def fit_auto_arima(series):
    if not HAS_AUTO_ARIMA:
        raise RuntimeError("auto_arima not available (pmdarima not installed).")
    model = auto_arima(
        series,
        start_p=0, start_q=0,
        max_p=4, max_q=4,
        d=None,
        seasonal=False,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    return model

def evaluate_forecast(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return mae, rmse, mape

def fetch_macro_data(start, end):
    macro_tickers = {
        "NIFTY 50": "^NSEI",
        "Crude Oil": "CL=F",
        "USDINR": "INR=X",
    }
    macro_data = {}
    for name, tick in macro_tickers.items():
        df = yf.download(tick, start=start, end=end, progress=False)
        if not df.empty:
            macro_data[name] = df["Close"].dropna()
    return macro_data

def corr_with_macro(price_series, macro_data):
    all_returns = {"Stock": price_series.pct_change()}
    for name, ser in macro_data.items():
        all_returns[name] = ser.pct_change()
    df = pd.DataFrame(all_returns).dropna()
    if df.empty:
        return pd.DataFrame()
    corr = df.corr().loc[["Stock"]]
    return corr.T

# ================== SIDEBAR – CONTROLS ==================

with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    ticker = st.text_input("NSE Ticker", value="RELIANCE.NS", help="Example: RELIANCE.NS, TCS.NS, HDFCBANK.NS")
    timeframe = st.selectbox(
        "Timeframe",
        ["2010–2018", "2021–2025", "Last 10 Years", "Max Available"],
    )
    freq = st.radio("Data Frequency", ["Monthly", "Daily"])
    st.markdown("---")
    model_choice = st.selectbox(
        "ARIMA Mode",
        ["Fixed ARIMA(1,1,1)", "Auto-ARIMA (select best p,d,q)"],
    )
    forecast_horizon = st.slider(
        "Forecast Horizon (steps ahead)",
        min_value=6,
        max_value=36,
        value=12,
        step=3,
        help="If Monthly → months ahead, if Daily → trading days ahead.",
    )
    test_size = st.slider(
        "Backtest Size (last N periods)",
        min_value=6,
        max_value=24,
        value=12,
        step=3,
        help="How many recent periods to keep aside for test / evaluation.",
    )
    st.markdown("---")
    st.caption("Hit **Analyze** to run full pipeline: history, indicators, ARIMA, metrics, residuals & macro view.")
    run_btn = st.button("🚀 Analyze")

# ================== HEADER ==================

st.markdown("<div class='main-title'>AuroraQuant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>ARIMA-driven stock lab – from history & indicators to model diagnostics & forecast quality.</div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)

# ================== MAIN FLOW ==================

if not run_btn:
    st.info("Configure settings in the **left sidebar** and click **Analyze** to start.")
    st.stop()

try:
    # ---------- set date range ----------
    if timeframe == "2010–2018":
        start, end = "2010-01-01", "2019-01-01"
    elif timeframe == "2021–2025":
        start, end = "2021-01-01", "2026-01-01"
    elif timeframe == "Last 10 Years":
        end = datetime.today().strftime("%Y-%m-%d")
        start_year = datetime.today().year - 10
        start = f"{start_year}-01-01"
    else:  # Max
        start, end = "2000-01-01", datetime.today().strftime("%Y-%m-%d")

    # ---------- fetch data ----------
    price = fetch_price_data(ticker, start, end, freq=freq)

    if len(price) < test_size + 12:
        st.warning("Not enough data for chosen test size. Reduce backtest periods or choose a longer timeframe.")
        st.stop()

    latest_price = float(price.iloc[-1])
    last_change_series = price.pct_change().iloc[-1] * 100
    last_change = float(last_change_series) if not np.isnan(last_change_series) else 0.0

    if len(price) > 252:
        high_52 = float(price.iloc[-252:].max())
        low_52 = float(price.iloc[-252:].min())
    else:
        high_52 = float(price.max())
        low_52 = float(price.min())

    # ---------- indicators ----------
    rsi = compute_rsi(price)
    macd, macd_sig, macd_hist = compute_macd(price)
    vol, returns = compute_volatility(price, freq=freq)

    # ================== MARKET PULSE CARDS ==================

    st.markdown("<div class='section-title'>Market Pulse</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-caption'>Instant snapshot of the stock based on the selected time window.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("<div class='info-title'>Current Snapshot</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-big'>₹ {latest_price:,.2f}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='metric-label'>Last move: {last_change:+.2f}%</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='metric-label'>52W High: ₹ {high_52:,.2f} &nbsp; | &nbsp; 52W Low: ₹ {low_52:,.2f}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("<div class='info-title'>Risk & Volatility</div>", unsafe_allow_html=True)
        if len(price) > 2:
            st.line_chart(price.tail(30))
        vol_val = vol.iloc[-1] if not np.isnan(vol.iloc[-1]) else np.nan
        vol_text = "Not enough data" if np.isnan(vol_val) else f"{vol_val*100:.2f}% (annualised)"
        st.markdown(
            f"<div class='metric-label'>Rolling volatility: {vol_text}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("<div class='info-title'>Momentum Snapshot</div>", unsafe_allow_html=True)
        latest_rsi = rsi.iloc[-1]
        rsi_level = 0.5 if np.isnan(latest_rsi) else min(max(latest_rsi / 100, 0.0), 1.0)
        st.progress(rsi_level)
        rsi_text = "N/A" if np.isnan(latest_rsi) else f"{latest_rsi:.2f}"
        st.markdown(
            f"<div class='metric-label'>RSI: {rsi_text}</div>",
            unsafe_allow_html=True,
        )
        macd_last = macd_hist.iloc[-1]
        macd_text = "N/A" if np.isnan(macd_last) else f"{macd_last:.4f}"
        st.markdown(
            f"<div class='metric-label'>MACD histogram (last): {macd_text}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)

    # ================== TIME MACHINE TABS ==================

    st.markdown("<div class='section-title'>Time Machine – History & Indicators</div>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(
        ["📈 Price Path", "📐 Technicals", "📊 Volatility & Volume"]
    )

    with t1:
        st.markdown("##### Price History")
        st.line_chart(price.rename("Close"))
        pct_change = price.pct_change() * 100
        st.markdown("##### Percentage Change")
        st.line_chart(pct_change.rename("% Change"))

    with t2:
        l, r = st.columns(2)
        with l:
            st.markdown("##### Price with Moving Averages")
            ma50 = price.rolling(50).mean()
            ma200 = price.rolling(200).mean()
            df_ma = pd.DataFrame({"Close": price, "MA 50": ma50, "MA 200": ma200})
            st.line_chart(df_ma)
        with r:
            st.markdown("##### RSI & MACD")
            st.line_chart(rsi.rename("RSI"))
            macd_df = pd.DataFrame({"MACD": macd, "Signal": macd_sig})
            st.line_chart(macd_df)

    with t3:
        l, r = st.columns(2)
        with l:
            st.markdown("##### Volatility")
            st.line_chart(vol.rename("Annualised Volatility"))
        with r:
            st.markdown("##### Volume (Daily)")
            daily = yf.download(ticker, start=start, end=end, progress=False)
            if not daily.empty and "Volume" in daily.columns:
                st.bar_chart(daily["Volume"].dropna())
            else:
                st.info("Volume not available for this ticker / range.")

    st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)

    # ================== ARIMA FORECASTING ==================

    st.markdown("<div class='section-title'>ARIMA Forecast Lab</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-caption'>Backtest on the last part of the series, then roll the same model forward to forecast.</div>",
        unsafe_allow_html=True,
    )

    # split train/test
    train = price.iloc[:-test_size]
    test = price.iloc[-test_size:]

    if model_choice.startswith("Fixed"):
        order = (1, 1, 1)
        arima_model = fit_fixed_arima(train, order)
        used_order = order
    else:
        if not HAS_AUTO_ARIMA:
            st.warning("Auto-ARIMA not available – falling back to ARIMA(1,1,1). Install pmdarima to enable.")
            order = (1, 1, 1)
            arima_model = fit_fixed_arima(train, order)
            used_order = order
        else:
            auto_model = fit_auto_arima(train)
            used_order = auto_model.order
            arima_model = auto_model  # use pmdarima model directly

    # forecast on test window
    if HAS_AUTO_ARIMA and isinstance(arima_model, type(auto_model)):
        test_forecast = pd.Series(arima_model.predict(n_periods=len(test)), index=test.index)
    else:
        test_forecast = arima_model.forecast(steps=len(test))
        test_forecast.index = test.index

    mae, rmse, mape = evaluate_forecast(test, test_forecast)

    # refit on full data for future forecast
    if model_choice.startswith("Fixed") or not HAS_AUTO_ARIMA:
        full_model = fit_fixed_arima(price, used_order)
        future_values = full_model.forecast(steps=forecast_horizon)
    else:
        auto_full = fit_auto_arima(price)
        used_order = auto_full.order
        future_values = pd.Series(
            auto_full.predict(n_periods=forecast_horizon),
        )

    last_index = price.index[-1]
    if freq == "Monthly":
        future_index = pd.date_range(last_index + pd.offsets.DateOffset(months=1),
                                     periods=forecast_horizon, freq="M")
    else:
        future_index = pd.date_range(last_index + pd.offsets.DateOffset(days=1),
                                     periods=forecast_horizon, freq="B")

    future_values.index = future_index

    # ---------- Forecast Tabs ----------
    f1, f2, f3 = st.tabs(
        ["🎯 Backtest: Model vs Actual", "🚀 Future Forecast", "🧪 Residual Diagnostics"]
    )

    with f1:
        st.markdown(f"##### Backtest – Last {test_size} periods (ARIMA{used_order})")
        df_bt = pd.DataFrame(
            {
                "Train": train,
                "Test (Actual)": test,
                "Forecast": test_forecast,
            }
        )
        st.line_chart(df_bt)
        st.markdown("###### Error Metrics")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("MAE", f"{mae:,.4f}")
        mcol2.metric("RMSE", f"{rmse:,.4f}")
        mcol3.metric("MAPE", f"{mape:,.2f}%")

    with f2:
        st.markdown("##### Historical + Forward ARIMA Path")
        df_future = pd.concat(
            [price.rename("Historical"), future_values.rename("Forecast")],
            axis=0,
        )
        st.line_chart(df_future)
        st.caption("Shaded area is not drawn, but conceptually represents forecast uncertainty widening over time.")

    with f3:
        st.markdown("##### Residual Check (Train Sample)")
        if model_choice.startswith("Fixed") or not HAS_AUTO_ARIMA:
            resid = arima_model.resid
        else:
            resid = pd.Series(auto_model.resid(), index=train.index)

        st.line_chart(resid.rename("Residuals"))
        st.caption("Ideally residuals should look like random noise around zero – no clear pattern / trend.")
        st.write("Residual summary:")
        st.write(resid.describe())

    st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)

    # ================== MACRO IMPACT VIEW ==================

    st.markdown("<div class='section-title'>Macro Impact Snapshot</div>", unsafe_allow_html=True)

    macro_data = fetch_macro_data(start, end)
    if macro_data:
        corr_df = corr_with_macro(price, macro_data)
        if not corr_df.empty:
            st.dataframe(
                corr_df.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                use_container_width=True,
            )
            st.caption("Correlation of stock returns with NIFTY, Crude Oil & USDINR over the same horizon.")
        else:
            st.info("Not enough overlapping data to compute macro correlations.")
    else:
        st.info("Macro data not available for this period.")

    st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)

    # ================== STRATEGY-STYLE SUMMARY ==================

    st.markdown("<div class='section-title'>Strategy-Style Summary (Educational Only)</div>", unsafe_allow_html=True)

    bullets = []

    ma50 = price.rolling(50).mean()
    ma200 = price.rolling(200).mean()
    if not ma50.dropna().empty and not ma200.dropna().empty:
        if ma50.iloc[-1] > ma200.iloc[-1]:
            bullets.append("Price above long-term MA → **structural uptrend bias**.")
        else:
            bullets.append("Price below long-term MA → **weak / corrective structure**.")

    if not np.isnan(latest_rsi):
        if latest_rsi > 70:
            bullets.append("RSI in overbought zone → **risk of short-term pullback**.")
        elif latest_rsi < 30:
            bullets.append("RSI in oversold zone → **scope for mean-reversion bounce**.")
        else:
            bullets.append("RSI neutral → **trend / news may dominate moves**.")

    if not np.isnan(vol_val):
        if vol_val > 0.4:
            bullets.append("High volatility → **suited only for aggressive risk-takers**.")
        elif vol_val < 0.15:
            bullets.append("Low volatility → **more stable, long-term friendly profile**.")
        else:
            bullets.append("Moderate volatility → **balanced risk-reward setting**.")

    bullets.append(
        f"Backtest MAPE ≈ **{mape:.2f}%** → ARIMA{used_order} captures some structure but still has forecast error, so use with caution."
    )

    for b in bullets:
        st.markdown(f"- {b}")

    st.caption("AuroraQuant is for learning ARIMA & market behaviour. **Not** a buy / sell recommendation.")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Check ticker, timeframe, internet, or try changing frequency / backtest size.")
