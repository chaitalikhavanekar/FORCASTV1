import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# ============= PAGE CONFIG & GLOBAL STYLE =============

st.set_page_config(
    page_title="AuroraQuant – Stock Time Machine",
    layout="wide",
)

# Custom CSS for better UI
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fb 0%, #ffffff 40%, #eef2ff 100%);
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    /* Center main title */
    .main-title {
        text-align: center;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin-bottom: 0rem;
    }
    .sub-title {
        text-align: center;
        font-size: 0.95rem;
        font-style: italic;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    /* Card style */
    .info-card {
        padding: 18px 20px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.86);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .info-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #4b5563;
        margin-bottom: 0.35rem;
    }
    .metric-big {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
    }
    .metric-label {
        font-size: 0.80rem;
        color: #6b7280;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.2rem 0 0.4rem 0;
        color: #111827;
    }
    .section-caption {
        font-size: 0.80rem;
        color: #6b7280;
        margin-bottom: 0.4rem;
    }
    .divider-soft {
        border-bottom: 1px solid rgba(148, 163, 184, 0.35);
        margin: 1.0rem 0 1.0rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============= HELPER FUNCTIONS =============

def fetch_price_data(ticker, start, end, freq="Monthly"):
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

def fit_arima(series, order=(1, 1, 1)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def safe_arima(series):
    for order in [(1,1,1), (2,1,1), (1,1,2)]:
        try:
            return fit_arima(series, order=order), order
        except Exception:
            continue
    raise ValueError("ARIMA model could not be fitted on this data.")

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
    all_returns = {}
    stock_ret = price_series.pct_change().rename("Stock")
    all_returns["Stock"] = stock_ret

    for name, ser in macro_data.items():
        all_returns[name] = ser.pct_change()

    df = pd.DataFrame(all_returns).dropna()
    if df.empty:
        return pd.DataFrame()
    corr = df.corr().loc[["Stock"]]
    return corr.T

# ============= SIDEBAR =============

with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    ticker_input = st.text_input(
        "NSE Ticker (e.g., RELIANCE.NS, TCS.NS)",
        value="RELIANCE.NS",
    )
    timeframe = st.selectbox(
        "Timeframe",
        ["2010–2018", "2021–2025", "Max Available"],
    )
    freq = st.radio("Data Frequency", ["Monthly", "Daily"])
    st.markdown("---")
    st.caption("Hit **Analyze** to pull data, compute indicators, and run ARIMA forecasting.")
    analyze = st.button("🚀 Analyze Stock")
    st.markdown("---")
    st.caption("AuroraQuant ∙ Experimental analytics only, **not investment advice**.")

# ============= HEADER =============

st.markdown("<div class='main-title'>AuroraQuant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Stock Time Machine & Forecast Lab – Explore history, indicators & ARIMA-powered forecasts.</div>",
    unsafe_allow_html=True,
)

st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)

# ============= MAIN LOGIC =============

if analyze:
    try:
        # ----- timeframe handling -----
        if timeframe == "2010–2018":
            start = "2010-01-01"
            end = "2019-01-01"
            test_year = 2018
            future_year = 2019
            future_periods = 12
        elif timeframe == "2021–2025":
            start = "2021-01-01"
            end = "2026-01-01"
            test_year = 2025
            future_year = 2026
            future_periods = 12
        else:
            start = "2000-01-01"
            end = datetime.today().strftime("%Y-%m-%d")
            last_year = datetime.today().year - 1
            test_year = last_year
            future_year = last_year + 1
            future_periods = 12

        # ----- fetch main series -----
        price = fetch_price_data(ticker_input, start, end, freq=freq)

        # robust scalar conversions (fixes Series.format error)
        latest_price = float(price.iloc[-1])
        last_change_val = price.pct_change().iloc[-1] * 100
        last_change = float(last_change_val) if not np.isnan(last_change_val) else 0.0

        if len(price) > 252:
            high_52 = float(price.iloc[-252:].max())
            low_52 = float(price.iloc[-252:].min())
        else:
            high_52 = float(price.max())
            low_52 = float(price.min())

        # indicators
        rsi = compute_rsi(price)
        macd, macd_signal, macd_hist = compute_macd(price)
        vol, returns = compute_volatility(price, freq=freq)

        # ============= MARKET PULSE (TOP CARDS) =============
        st.markdown("<div class='section-title'>Market Pulse</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-caption'>Quick snapshot of price, risk, and momentum based on the selected period & frequency.</div>",
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
            if np.isnan(vol_val):
                vol_text = "Not enough data"
            else:
                vol_text = f"{vol_val*100:.2f}% (annualised)"
            st.markdown(
                f"<div class='metric-label'>Rolling volatility: {vol_text}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with c3:
            st.markdown("<div class='info-card'>", unsafe_allow_html=True)
            st.markdown("<div class='info-title'>Momentum Signal</div>", unsafe_allow_html=True)
            latest_rsi = rsi.iloc[-1]
            if np.isnan(latest_rsi):
                rsi_level = 0.5
            else:
                rsi_level = latest_rsi / 100
            st.progress(min(max(rsi_level, 0.0), 1.0))
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

        # ============= TIME MACHINE – HISTORY & INDICATORS =============
        st.markdown("<div class='section-title'>Time Machine – Historical View</div>", unsafe_allow_html=True)

        tm_tab1, tm_tab2, tm_tab3 = st.tabs(
            ["📈 Price Universe", "📐 Technical Layer", "📊 Volatility & Volume"]
        )

        with tm_tab1:
            st.markdown("##### Historical Price")
            st.line_chart(price.rename("Close"))
            st.caption("Base price trend across the selected horizon.")

        with tm_tab2:
            left, right = st.columns(2)

            with left:
                st.markdown("##### Price with Moving Averages (50 & 200)")
                df_ma = pd.DataFrame(
                    {
                        "Close": price,
                        "MA 50": price.rolling(window=50).mean(),
                        "MA 200": price.rolling(window=200).mean(),
                    }
                )
                st.line_chart(df_ma)

            with right:
                st.markdown("##### RSI & MACD")
                st.line_chart(rsi.rename("RSI"))
                macd_df = pd.DataFrame({"MACD": macd, "Signal": macd_signal})
                st.line_chart(macd_df)
                st.caption("RSI > 70 → overbought, RSI < 30 → oversold. MACD crossovers show momentum shifts.")

        with tm_tab3:
            left, right = st.columns(2)
            with left:
                st.markdown("##### Volume (Daily)")
                daily_data = yf.download(ticker_input, start=start, end=end, progress=False)
                if not daily_data.empty and "Volume" in daily_data.columns:
                    st.bar_chart(daily_data["Volume"].dropna())
                else:
                    st.info("Volume data not available for this ticker/period.")
            with right:
                st.markdown("##### Rolling Volatility")
                st.line_chart(vol.rename("Annualised Volatility"))

        st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)

        # ============= FORECAST LAB – ARIMA =============
        st.markdown("<div class='section-title'>Forecast Lab – ARIMA & Macro Lens</div>", unsafe_allow_html=True)

        fc_tab1, fc_tab2, fc_tab3 = st.tabs(
            ["🎯 Model vs Reality", "🚀 Time Tunnel Forecast", "🌐 Macro Impact Radar"]
        )

        # prepare train/test
        train = price[price.index < f"{test_year}-01-01"]
        test = price[price.index >= f"{test_year}-01-01"]

        if len(train) < 12:
            st.warning("Not enough data for ARIMA in this timeframe. Try 'Max Available'.")
        else:
            model_fit, used_order = safe_arima(train)
            forecast_test = model_fit.forecast(steps=len(test))
            forecast_test.index = test.index

            model_full, used_order_full = safe_arima(price)
            future_forecast = model_full.forecast(steps=future_periods)
            future_index = pd.date_range(
                start=price.index[-1]
                + (pd.offsets.DateOffset(months=1) if freq == "Monthly" else pd.offsets.DateOffset(days=1)),
                periods=future_periods,
                freq=("M" if freq == "Monthly" else "D"),
            )
            future_forecast.index = future_index

            with fc_tab1:
                st.markdown(f"##### Train vs Test vs Forecast (ARIMA{used_order}, Test Year {test_year})")
                df_fc = pd.DataFrame(
                    {
                        "Train": train,
                        "Test (Actual)": test,
                        "Forecast (Test)": forecast_test,
                    }
                )
                st.line_chart(df_fc)
                st.caption("Left: training data. Right: test year with ARIMA forecast overlapped on actual prices.")

            with fc_tab2:
                st.markdown(f"##### Future Path for {future_year}")
                df_future = pd.concat(
                    [
                        price.rename("Historical"),
                        future_forecast.rename("Forecast"),
                    ],
                    axis=0,
                )
                st.line_chart(df_future)
                st.caption("Forecast is purely model-based and for academic / experimental use only.")

            with fc_tab3:
                st.markdown("##### Macro Correlation Snapshot")
                macro_data = fetch_macro_data(start, end)
                if macro_data:
                    corr_df = corr_with_macro(price, macro_data)
                    if not corr_df.empty:
                        st.dataframe(
                            corr_df.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                            use_container_width=True,
                        )
                        st.caption(
                            "Correlation of stock returns with NIFTY, Crude Oil, and USDINR over the same horizon."
                        )
                    else:
                        st.info("Not enough overlapping data to compute correlations.")
                else:
                    st.info("Macro data could not be fetched for this horizon.")

        st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)

        # ============= STRATEGY SNAPSHOT =============
        st.markdown("<div class='section-title'>Strategy Snapshot (Conceptual)</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-caption'>A human-readable summary combining trend, momentum, and risk. "
            "Use it only as an educational signal, not as trading advice.</div>",
            unsafe_allow_html=True,
        )

        bullets = []

        ma50 = price.rolling(window=50).mean()
        ma200 = price.rolling(window=200).mean()
        if not ma50.dropna().empty and not ma200.dropna().empty:
            if ma50.iloc[-1] > ma200.iloc[-1]:
                bullets.append("Price is above long-term moving average → **structural uptrend bias.**")
            else:
                bullets.append("Price is below long-term moving average → **weak / corrective phase.**")

        latest_rsi = rsi.iloc[-1]
        if not np.isnan(latest_rsi):
            if latest_rsi > 70:
                bullets.append("RSI in overbought zone → **risk of short-term pullback.**")
            elif latest_rsi < 30:
                bullets.append("RSI in oversold zone → **scope for relief rally / mean reversion.**")
            else:
                bullets.append("RSI neutral → **no extreme momentum; trend-driven moves.**")

        vol_val = vol.iloc[-1]
        if not np.isnan(vol_val):
            if vol_val > 0.4:
                bullets.append("Volatility is high → **suited only to aggressive, risk-tolerant traders.**")
            elif vol_val < 0.15:
                bullets.append("Volatility is low → **more stable, long-term friendly profile.**")
            else:
                bullets.append("Volatility is moderate → **balanced risk / reward setup.**")

        if bullets:
            for b in bullets:
                st.markdown(f"- {b}")
        else:
            st.markdown("- Not enough data to generate a meaningful summary.")

        st.caption("AuroraQuant is an academic tool. This is **not** a buy/sell recommendation.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Check the ticker symbol, internet connection, or try another timeframe / frequency.")

else:
    st.info("Adjust settings in the **left sidebar**, then click **Analyze Stock** to start.")
