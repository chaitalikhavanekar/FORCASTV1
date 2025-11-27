import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# ============= BASIC CONFIG =============
st.set_page_config(
    page_title="AuroraQuant – Stock Time Machine",
    layout="wide",
)

# ============= HELPER FUNCTIONS =============

def fetch_price_data(ticker, start, end, freq="Monthly"):
    """Fetch price data from Yahoo Finance and resample."""
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
    """Try a few ARIMA orders to avoid crashes."""
    for order in [(1,1,1), (2,1,1), (1,1,2)]:
        try:
            return fit_arima(series, order=order), order
        except Exception:
            continue
    raise ValueError("ARIMA model could not be fitted on this data.")

def fetch_macro_data(start, end):
    """Fetch simple macro series for correlation: NIFTY, Crude, USDINR."""
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
    """Return correlation of stock returns with macro returns."""
    all_returns = {}
    stock_ret = price_series.pct_change().rename("Stock")
    all_returns["Stock"] = stock_ret

    for name, ser in macro_data.items():
        all_returns[name] = ser.pct_change()

    df = pd.DataFrame(all_returns).dropna()
    if df.empty:
        return pd.DataFrame()
    corr = df.corr().loc[["Stock"]]
    return corr.T  # to show as column

# ============= UI LAYOUT =============

st.markdown(
    "<h1 style='text-align: center;'>AuroraQuant</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; font-style: italic;'>Stock Time Machine & Forecast Lab</p>",
    unsafe_allow_html=True,
)

col_top1, col_top2, col_top3 = st.columns([2, 1, 1])

with col_top1:
    ticker_input = st.text_input("Enter NSE Ticker (e.g., RELIANCE.NS, TCS.NS)", value="RELIANCE.NS")

with col_top2:
    timeframe = st.selectbox("Timeframe", ["2010–2018", "2021–2025", "Max Available"])

with col_top3:
    freq = st.radio("Frequency", ["Monthly", "Daily"], horizontal=True)

analyze = st.button("🚀 Analyze")

st.markdown("---")

if analyze:
    try:
        # --------- Determine date range based on timeframe ---------
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
        else:  # Max Available
            start = "2000-01-01"
            end = datetime.today().strftime("%Y-%m-%d")
            # last complete year as test year
            last_year = datetime.today().year - 1
            test_year = last_year
            future_year = last_year + 1
            future_periods = 12

        # --------- FETCH MAIN PRICE DATA ---------
        price = fetch_price_data(ticker_input, start, end, freq=freq)
        latest_price = float(price.iloc[-1])
        last_change = price.pct_change().iloc[-1] * 100
        high_52 = price.iloc[-252:].max() if len(price) > 252 else price.max()
        low_52 = price.iloc[-252:].min() if len(price) > 252 else price.min()

        # --------- INDICATORS ---------
        rsi = compute_rsi(price)
        macd, macd_signal, macd_hist = compute_macd(price)
        vol, returns = compute_volatility(price, freq=freq)

        # ============= MARKET PULSE =============
        st.subheader("Market Pulse")

        pulse_col1, pulse_col2, pulse_col3 = st.columns(3)

        with pulse_col1:
            st.markdown("**Current Snapshot**")
            st.metric("Last Close", f"₹ {latest_price:,.2f}", f"{last_change:+.2f}%")
            st.caption(f"Approx 52W High: ₹ {high_52:,.2f} | 52W Low: ₹ {low_52:,.2f}")

        with pulse_col2:
            st.markdown("**Risk & Volatility**")
            # Sparkline: last 30 periods
            if len(price) > 2:
                st.line_chart(price.tail(30))
            st.caption(f"Rolling Volatility (last point): {vol.iloc[-1]*100:.2f}%" if not np.isnan(vol.iloc[-1]) else "Volatility: Not enough data")

        with pulse_col3:
            st.markdown("**Momentum Signal**")
            latest_rsi = rsi.iloc[-1]
            rsi_level = 0.5 if np.isnan(latest_rsi) else latest_rsi / 100
            st.progress(min(max(rsi_level, 0.0), 1.0))
            st.caption(f"RSI: {latest_rsi:.2f}")
            st.caption("MACD Histogram (last): " + (f"{macd_hist.iloc[-1]:.4f}" if not np.isnan(macd_hist.iloc[-1]) else "N/A"))

        st.markdown("---")

        # ============= TIME MACHINE – HISTORICAL VIEW =============
        st.subheader("Time Machine – Historical View")

        tm_tab1, tm_tab2, tm_tab3 = st.tabs(["📈 Price Universe", "📐 Technical Layer", "📊 Volatility & Volume"])

        with tm_tab1:
            st.markdown("**Historical Price Chart**")
            st.line_chart(price)
            st.caption("You can interpret this as the base trend / price movement over the selected horizon.")

        with tm_tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Price with Moving Averages (50 & 200)**")
                df_ma = pd.DataFrame({
                    "Close": price,
                    "MA 50": price.rolling(window=50).mean(),
                    "MA 200": price.rolling(window=200).mean(),
                })
                st.line_chart(df_ma)
                st.caption("Golden Cross / Death Cross zones can be discussed based on MA 50 vs MA 200.")

            with c2:
                st.markdown("**RSI & MACD**")
                st.line_chart(rsi.rename("RSI"))
                macd_df = pd.DataFrame({
                    "MACD": macd,
                    "Signal": macd_signal
                })
                st.line_chart(macd_df)
                st.caption("RSI above 70 ~ overbought; below 30 ~ oversold. MACD crossovers indicate momentum shifts.")

        with tm_tab3:
            st.markdown("**Volume & Volatility**")
            v1, v2 = st.columns(2)
            with v1:
                st.markdown("Volume (if available at original frequency)")
                # volume only available at original frequency – so re-download daily
                daily_data = yf.download(ticker_input, start=start, end=end, progress=False)
                if not daily_data.empty:
                    vol_series = daily_data["Volume"].dropna()
                    st.bar_chart(vol_series)
                else:
                    st.info("Volume data not available.")
            with v2:
                st.markdown("Rolling Volatility")
                st.line_chart(vol.rename("Volatility"))
                st.caption("Volatility is annualized rolling standard deviation of returns.")

        st.markdown("---")

        # ============= FORECAST LAB – ARIMA & BEYOND =============
        st.subheader("Forecast Lab – ARIMA & Beyond")

        fc_tab1, fc_tab2, fc_tab3 = st.tabs(["🎯 Model vs Reality", "🚀 Time Tunnel Forecast", "🌐 Macro Impact Radar"])

        # --------- Train-Test Split for ARIMA ---------
        train = price[price.index < f"{test_year}-01-01"]
        test = price[price.index >= f"{test_year}-01-01"]

        if len(train) < 12:
            st.warning("Not enough data for a proper ARIMA model in this timeframe. Try 'Max Available'.")
        else:
            model_fit, used_order = safe_arima(train)
            forecast_test = model_fit.forecast(steps=len(test))
            forecast_test.index = test.index

            # Refit on full series for future forecast
            model_full, used_order_full = safe_arima(price)
            future_forecast = model_full.forecast(steps=future_periods)
            future_index = pd.date_range(
                start=price.index[-1] + pd.offsets.DateOffset(months=1 if freq=="Monthly" else 1),
                periods=future_periods,
                freq=("M" if freq=="Monthly" else "D")
            )
            future_forecast.index = future_index

            with fc_tab1:
                st.markdown(f"**Train vs Test vs Forecast – ARIMA{used_order} – Test Year: {test_year}**")
                df_fc = pd.DataFrame({
                    "Train": train,
                    "Test (Actual)": test,
                    "Forecast (Test)": forecast_test
                })
                st.line_chart(df_fc)
                st.caption("Left side = training data, right side = test period with ARIMA forecast overlapped.")

            with fc_tab2:
                st.markdown(f"**Future Path – ARIMA Forecast for {future_year}**")
                df_future = pd.concat([price.rename("Historical"), future_forecast.rename("Forecast")], axis=0)
                st.line_chart(df_future)
                st.caption("Forecast is extrapolated based on historical pattern. This is purely for academic / experimental use.")

            with fc_tab3:
                st.markdown("**Macro Correlation Heatmap (Simple)**")
                macro_data = fetch_macro_data(start, end)
                if macro_data:
                    corr_df = corr_with_macro(price, macro_data)
                    if not corr_df.empty:
                        st.dataframe(corr_df.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1))
                        st.caption("Correlation of stock returns with macro factors (NIFTY, Crude, USDINR).")
                    else:
                        st.info("Not enough overlapping data for correlation.")
                else:
                    st.info("Macro data could not be fetched for this period.")

        st.markdown("---")

        # ============= STRATEGY PANEL =============
        st.subheader("Strategy Snapshot (Conceptual, Not Investment Advice)")

        # Quick text summary based on a few indicators
        summary_points = []

        # Trend via MA
        ma50 = price.rolling(window=50).mean()
        ma200 = price.rolling(window=200).mean()
        if not ma50.dropna().empty and not ma200.dropna().empty:
            if ma50.iloc[-1] > ma200.iloc[-1]:
                summary_points.append("Price is above long-term moving average → **Uptrend bias**.")
            else:
                summary_points.append("Price is below long-term moving average → **Downtrend / weak trend**.")

        # RSI
        if not np.isnan(rsi.iloc[-1]):
            if rsi.iloc[-1] > 70:
                summary_points.append("RSI is in overbought zone → **Possibility of correction / profit booking.**")
            elif rsi.iloc[-1] < 30:
                summary_points.append("RSI is in oversold zone → **Possibility of bounce / mean reversion.**")
            else:
                summary_points.append("RSI is in neutral zone → **No extreme momentum.**")

        # Volatility
        if not np.isnan(vol.iloc[-1]):
            if vol.iloc[-1] > 0.4:
                summary_points.append("Volatility is high → **Risky, suitable only for aggressive traders.**")
            elif vol.iloc[-1] < 0.15:
                summary_points.append("Volatility is low → **Stable, may suit conservative investors.**")
            else:
                summary_points.append("Volatility is moderate → **Balanced risk profile.**")

        if summary_points:
            for s in summary_points:
                st.markdown(f"- {s}")
        else:
            st.markdown("- Not enough data to summarize indicators.")

        st.caption("This is an educational / analytical tool. Not a recommendation to buy / sell.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Check ticker symbol, internet connection, or try a different timeframe.")
else:
    st.info("Enter a stock ticker and click **Analyze** to start the Time Machine.")
