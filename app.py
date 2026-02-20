"""
Statistical Arbitrage Engine — Streamlit Dashboard
===================================================

Run with:
    streamlit run app.py

The dashboard has four tabs:
  1. Pair Scanner   — run cointegration tests and explore results
  2. Signal Monitor — live z-score chart and current positions
  3. Backtest       — run and analyse a single-pair backtest
  4. Risk Report    — portfolio-level risk analytics
"""

import sys
from pathlib import Path

# Make src importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.backtesting.engine import BacktestConfig, PairsBacktestEngine
from src.data.data_loader import SP500_UNIVERSE, DataLoader
from src.risk.risk_manager import RiskAnalytics, compute_portfolio_heatmap, rolling_sharpe
from src.signals.cointegration import PairScanner, compute_spread, estimate_half_life
from src.signals.kalman_filter import KalmanFilterHedge, MeanReversionSignal
from src.utils.plotting import (
    plot_correlation_heatmap,
    plot_drawdown,
    plot_equity_curve,
    plot_kalman_hedge_ratio,
    plot_performance_summary,
    plot_return_distribution,
    plot_spread_and_signals,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Statistical Arbitrage Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — global settings
# ---------------------------------------------------------------------------

st.sidebar.title("Statistical Arbitrage Engine")
st.sidebar.markdown("---")

st.sidebar.subheader("Data Settings")
start_date = st.sidebar.date_input("Start date", value=pd.Timestamp("2020-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.Timestamp("2024-12-31"))

st.sidebar.subheader("Cointegration Settings")
eg_threshold = st.sidebar.slider("Engle-Granger p-value threshold", 0.01, 0.10, 0.05, 0.01)
min_hl = st.sidebar.slider("Min half-life (days)", 2, 20, 5)
max_hl = st.sidebar.slider("Max half-life (days)", 20, 120, 60)

st.sidebar.subheader("Signal Settings")
entry_z = st.sidebar.slider("Entry z-score", 1.0, 3.0, 2.0, 0.1)
exit_z = st.sidebar.slider("Exit z-score", 0.1, 1.5, 0.5, 0.1)
stop_z = st.sidebar.slider("Stop-loss z-score", 2.0, 5.0, 3.5, 0.1)
zscore_window = st.sidebar.slider("Z-score rolling window (days)", 10, 60, 30)

st.sidebar.subheader("Backtest Settings")
initial_capital = st.sidebar.number_input("Initial capital ($)", value=1_000_000, step=100_000)
position_size_pct = st.sidebar.slider("Position size (% of capital)", 0.05, 0.30, 0.10, 0.01)
tx_cost_bps = st.sidebar.slider("Transaction cost (bps)", 1, 20, 5)
slippage_bps = st.sidebar.slider("Slippage (bps)", 0, 10, 2)

# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner="Downloading price data from Yahoo Finance…")
def load_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    loader = DataLoader(start_date=start, end_date=end)
    return loader.fetch_multiple_tickers(tickers)


@st.cache_data(ttl=3600, show_spinner="Running cointegration scan…")
def run_cointegration_scan(
    prices_json: str, eg_thr: float, min_half: float, max_half: float
) -> pd.DataFrame:
    prices = pd.read_json(prices_json)
    prices.index = pd.to_datetime(prices.index)
    scanner = PairScanner(
        prices,
        eg_pvalue_threshold=eg_thr,
        min_half_life=min_half,
        max_half_life=max_half,
    )
    report = scanner.scan()
    return report.to_dataframe()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["Pair Scanner", "Signal Monitor", "Backtest", "Risk Report"]
)

# ===========================================================================
# TAB 1: Pair Scanner
# ===========================================================================

with tab1:
    st.header("Pair Scanner — Cointegration Analysis")
    st.markdown(
        """
        **How it works:**
        1. Download adjusted close prices for all tickers.
        2. Test every unique pair with **Engle-Granger** and **Johansen** cointegration tests.
        3. Estimate the **half-life** of mean reversion for each pair.
        4. Keep only pairs that pass all three filters.
        """
    )

    selected_tickers = st.multiselect(
        "Select tickers to include in the universe",
        options=SP500_UNIVERSE,
        default=SP500_UNIVERSE[:20],
    )

    if st.button("Run Pair Scanner", type="primary") and len(selected_tickers) >= 2:
        with st.spinner("Loading prices…"):
            prices = load_prices(selected_tickers, str(start_date), str(end_date))

        st.success(f"Loaded {prices.shape[1]} tickers × {prices.shape[0]} trading days.")

        with st.spinner("Testing all pairs for cointegration…"):
            results_df = run_cointegration_scan(
                prices.to_json(), eg_threshold, min_hl, max_hl
            )

        coint_df = results_df[results_df["is_cointegrated"]]
        st.metric("Cointegrated pairs found", len(coint_df))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("All cointegrated pairs")
            st.dataframe(
                coint_df[["pair", "eg_pvalue", "johansen_trace", "hedge_ratio", "half_life_days"]]
                .rename(columns={"half_life_days": "half_life"}),
                use_container_width=True,
            )
        with col2:
            st.subheader("P-value distribution (all pairs)")
            import plotly.express as px
            fig = px.histogram(
                results_df, x="eg_pvalue", nbins=40,
                title="Engle-Granger P-value Distribution",
                labels={"eg_pvalue": "p-value"},
                template="plotly_white",
            )
            fig.add_vline(x=eg_threshold, line_dash="dash", line_color="red",
                          annotation_text=f"threshold={eg_threshold}")
            st.plotly_chart(fig, use_container_width=True)

        # Cache results for other tabs
        st.session_state["prices"] = prices
        st.session_state["coint_pairs"] = coint_df

    elif len(selected_tickers) < 2:
        st.warning("Select at least 2 tickers.")

# ===========================================================================
# TAB 2: Signal Monitor
# ===========================================================================

with tab2:
    st.header("Signal Monitor — Live Z-Score & Kalman Filter")

    if "prices" not in st.session_state or "coint_pairs" not in st.session_state:
        st.info("Run the Pair Scanner first to populate this tab.")
    else:
        prices = st.session_state["prices"]
        coint_df = st.session_state["coint_pairs"]

        if coint_df.empty:
            st.warning("No cointegrated pairs found. Adjust the scanner settings.")
        else:
            pair_options = coint_df["pair"].tolist()
            selected_pair = st.selectbox("Select a pair to monitor", pair_options)

            row = coint_df[coint_df["pair"] == selected_pair].iloc[0]
            ticker_a, ticker_b = row["ticker_a"], row["ticker_b"]

            if ticker_a in prices.columns and ticker_b in prices.columns:
                pa = prices[ticker_a].dropna()
                pb = prices[ticker_b].dropna()

                kf = KalmanFilterHedge(zscore_window=zscore_window)
                kf_output = kf.fit(pa, pb)

                signal_gen = MeanReversionSignal(
                    entry_z=entry_z, exit_z=exit_z, stop_z=stop_z
                )
                signals = signal_gen.generate(kf_output)

                # Current state
                latest = kf_output.iloc[-1]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Z-Score", f"{latest['zscore']:.2f}")
                col2.metric("Hedge Ratio β", f"{latest['beta']:.3f}")
                col3.metric("Current Signal", {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(int(signals.iloc[-1]), "FLAT"))
                col4.metric("Spread (log)", f"{latest['spread']:.4f}")

                # Charts
                st.plotly_chart(
                    plot_spread_and_signals(
                        kf_output["spread"], kf_output["zscore"], signals,
                        entry_z=entry_z, exit_z=exit_z,
                        title=f"{selected_pair} — Spread & Signals",
                    ),
                    use_container_width=True,
                )
                st.plotly_chart(
                    plot_kalman_hedge_ratio(kf_output["beta"], title=f"{selected_pair} — Kalman Hedge Ratio"),
                    use_container_width=True,
                )

# ===========================================================================
# TAB 3: Backtest
# ===========================================================================

with tab3:
    st.header("Backtest — Realistic Simulation")
    st.markdown(
        """
        Simulates trading with:
        - **Next-day execution** (no look-ahead bias)
        - **Transaction costs** and **slippage** deducted on every trade
        - **Stop-loss** and **maximum holding period** enforced
        """
    )

    if "prices" not in st.session_state or "coint_pairs" not in st.session_state:
        st.info("Run the Pair Scanner first.")
    else:
        prices = st.session_state["prices"]
        coint_df = st.session_state["coint_pairs"]

        if coint_df.empty:
            st.warning("No cointegrated pairs to backtest.")
        else:
            pair_options = coint_df["pair"].tolist()
            bt_pair = st.selectbox("Select pair to backtest", pair_options, key="bt_pair")

            if st.button("Run Backtest", type="primary"):
                row = coint_df[coint_df["pair"] == bt_pair].iloc[0]
                ta, tb = row["ticker_a"], row["ticker_b"]

                pa = prices[ta].dropna()
                pb = prices[tb].dropna()

                kf = KalmanFilterHedge(zscore_window=zscore_window)
                kf_output = kf.fit(pa, pb)

                sig_gen = MeanReversionSignal(entry_z=entry_z, exit_z=exit_z, stop_z=stop_z)
                signals = sig_gen.generate(kf_output)

                cfg = BacktestConfig(
                    initial_capital=float(initial_capital),
                    position_size_pct=position_size_pct,
                    transaction_cost_bps=tx_cost_bps,
                    slippage_bps=slippage_bps,
                )
                engine = PairsBacktestEngine(config=cfg)
                metrics = engine.run(
                    price_a=pa,
                    price_b=pb,
                    signals=signals,
                    hedge_ratio=kf_output["beta"],
                    pair_name=bt_pair,
                    zscore=kf_output["zscore"],
                )

                # Store for risk tab
                st.session_state["bt_metrics"] = metrics

                m = metrics.to_dict()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}",
                            delta="✓" if m['sharpe_ratio'] > 1.5 else "✗")
                col2.metric("Max Drawdown", f"{m['max_drawdown_pct']:.1f}%",
                            delta="✓" if abs(m['max_drawdown_pct']) < 15 else "✗")
                col3.metric("Win Rate", f"{m['win_rate_pct']:.1f}%",
                            delta="✓" if m['win_rate_pct'] > 55 else "✗")
                col4.metric("# Trades", m['n_trades'])

                st.plotly_chart(
                    plot_equity_curve(metrics.equity_curve, title=f"{bt_pair} — Equity Curve"),
                    use_container_width=True,
                )

                col_left, col_right = st.columns(2)
                with col_left:
                    st.plotly_chart(
                        plot_drawdown(metrics.daily_returns, title="Drawdown"),
                        use_container_width=True,
                    )
                with col_right:
                    st.plotly_chart(
                        plot_return_distribution(metrics.daily_returns),
                        use_container_width=True,
                    )

                st.plotly_chart(
                    plot_performance_summary(m),
                    use_container_width=True,
                )

                # Trade log
                if metrics.trades:
                    trade_rows = []
                    for t in metrics.trades:
                        if t.exit_date:
                            trade_rows.append({
                                "Entry": t.entry_date.date() if t.entry_date else "",
                                "Exit": t.exit_date.date(),
                                "Direction": {1: "Long", -1: "Short"}.get(t.direction, ""),
                                "P&L ($)": round(t.pnl, 2),
                                "Return (%)": round(t.return_pct * 100, 2),
                                "Holding Days": t.holding_days,
                                "Exit Reason": t.exit_reason,
                            })
                    st.subheader("Trade Log")
                    st.dataframe(pd.DataFrame(trade_rows), use_container_width=True)

# ===========================================================================
# TAB 4: Risk Report
# ===========================================================================

with tab4:
    st.header("Risk Report")

    if "bt_metrics" not in st.session_state:
        st.info("Run a backtest first to populate this tab.")
    else:
        metrics = st.session_state["bt_metrics"]
        analytics = RiskAnalytics(metrics.daily_returns)
        summary = analytics.summary()

        st.subheader("Risk Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Sharpe Ratio", summary["sharpe_ratio"])
        col1.metric("Sortino Ratio", summary["sortino_ratio"])
        col1.metric("Calmar Ratio", summary["calmar_ratio"])
        col2.metric("Ann. Return", f"{summary['annualised_return_pct']}%")
        col2.metric("Ann. Volatility", f"{summary['annualised_volatility_pct']}%")
        col2.metric("Max Drawdown", f"{summary['max_drawdown_pct']}%")
        col3.metric("95% VaR (1-day)", f"{summary['var_95_pct']}%")
        col3.metric("95% CVaR (1-day)", f"{summary['cvar_95_pct']}%")
        col3.metric("Trading Days", summary["n_trading_days"])

        # Rolling Sharpe
        roll_sh = rolling_sharpe(metrics.daily_returns, window=63)
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(
            x=roll_sh.index, y=roll_sh.values,
            line=dict(color="#2563eb", width=2),
            name="Rolling Sharpe (63-day)",
        ))
        fig.add_hline(y=1.5, line_dash="dash", line_color="green",
                      annotation_text="Target (1.5)")
        fig.update_layout(
            title="Rolling Sharpe Ratio (1 Quarter)",
            xaxis_title="Date", yaxis_title="Sharpe",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.plotly_chart(
            plot_drawdown(metrics.daily_returns, title="Portfolio Drawdown"),
            use_container_width=True,
        )
