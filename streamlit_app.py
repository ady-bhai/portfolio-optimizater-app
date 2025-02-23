import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import cvxpy as cp
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA  # Replacing TensorFlow


def main():
    st.set_page_config(page_title="Portfolio Optimizer Pro", layout="wide")
    st.title("Portfolio Optimizer")
    st.markdown("""
    ## Institutional-Grade Portfolio Construction
    """)

    # ======================
    # Sidebar Controls
    # ======================
    with st.sidebar:
        st.header("⚙️ Control Panel")
        delta = st.slider("Risk Tolerance (δ)", 0.01, 1.0, 0.15, 
                         help="Higher = More aggressive portfolio")
        tau = st.slider("Market Trust (τ)", 0.01, 0.5, 0.05,
                       help="Confidence in market equilibrium vs. your views")
        min_weight = st.slider("Min Allocation", 0.0, 0.3, 0.05)
        max_weight = st.slider("Max Allocation", 0.1, 0.5, 0.25)
        lambda_tc = st.slider("Trading Cost Factor", 0.0001, 0.01, 0.001)
        
        st.markdown("---")
        with st.expander("ℹ️ View Setup Guide"):
            st.markdown("""
            **Absolute View Example:**  
            "I expect SPY to return 8% annually with 70% confidence"  
            → Set Return = 8%, Confidence = 0.7  

            **Relative View Example:**  
            "QQQ will beat TLT by 3% with 50% confidence"  
            → Select QQQ/Underperformer=TLT, Spread=3%, Confidence=0.5  
            """)

    # ======================
    # Data Input Section
    # ======================
    st.header("📊 Asset Setup")
    default_tickers = ["SPY", "QQQ", "TLT", "GLD", "VNQ", "BTC-USD"]
    tickers = st.multiselect("Select assets (min 5)", default_tickers, default_tickers)
    
    # Load data with fallback
    @st.cache_data
    def load_data(tickers):
        try:
            df = yf.download(tickers, period="5y")['Adj Close']
            # Handle single ticker case
            if len(tickers) == 1:
                df = pd.DataFrame(df).rename(columns={'Adj Close': tickers[0]})
            return df.ffill().dropna()
        except Exception as e:
            st.error(f"Data loading failed: {str(e)}")
            # Create synthetic data for demonstration
            dates = pd.date_range(end=pd.Timestamp.today(), periods=1260)
            return pd.DataFrame(
                np.random.normal(100, 5, (1260, len(tickers))),
                index=dates,
                columns=tickers
            )
    prices = load_data(tickers)
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


    # ======================
    # View Configuration
    # ======================
    st.header("🎯 Investment Views")
    P, Q, confidences = [], [], []
    num_assets = len(tickers)

    with st.expander("🔍 Absolute Views", expanded=True):
        cols = st.columns(3)
        for i, ticker in enumerate(tickers):
            with cols[i%3]:
                if st.checkbox(f"Set view for {ticker}", key=f"abs_{ticker}"):
                    view_return = st.number_input(f"{ticker} excess return (%)", 
                                                 value=5.0, key=f"ret_{ticker}")/100
                    confidence = st.slider(f"Confidence {ticker}", 0.1, 1.0, 0.7,
                                          key=f"conf_{ticker}")
                    P.append(np.eye(1, num_assets, i)[0])
                    Q.append(view_return)
                    confidences.append(confidence)

    with st.expander("📈 Relative Views"):
        col1, col2 = st.columns(2)
        with col1:
            outperformer = st.selectbox("Outperforming asset", tickers)
        with col2:
            underperformer = st.selectbox("Underperforming asset", 
                                        [t for t in tickers if t != outperformer])
        spread = st.number_input("Performance spread (%)", value=3.0)/100
        conf_rel = st.slider("View confidence", 0.1, 1.0, 0.5)
        
        if st.button("Add Relative View"):
            i = tickers.index(outperformer)
            j = tickers.index(underperformer)
            p_vec = np.zeros(num_assets)
            p_vec[i] = 1
            p_vec[j] = -1
            P.append(p_vec)
            Q.append(spread)
            confidences.append(conf_rel)

    # ======================
    # Validation Checks
    # ======================
    if not P:
        st.error("⚠️ Add at least one view before optimizing!")
        st.stop()
        
    if min_weight * len(tickers) > 1:
        st.error(f"Impossible: {len(tickers)} assets × min {min_weight} > 100%")
        st.stop()

    # ======================
    # Covariance Estimation
    # ======================
    def ledoit_wolf_shrinkage(returns):
        X = returns.values
        n, p = X.shape
        S = np.cov(X, rowvar=False)
        F = np.mean(S.diagonal()) * np.eye(p)  # Improved target
        delta = max(0, min(1, (np.trace(S@S) + np.trace(S)**2) / 
                          (n * np.trace((S - F)**2))))
        return delta * F + (1 - delta) * S

    S = ledoit_wolf_shrinkage(returns)

    # AI-Free Forecasting Section
    # ======================
    st.header("📈 Statistical Forecast Engine")

    def forecast_returns(prices, days=30):
        forecasts = {}
        for ticker in tickers:
            model = ARIMA(prices[ticker].dropna(), order=(1, 0, 0))
            model_fit = model.fit()
            pred = model_fit.forecast(steps=days).mean()
            forecasts[ticker] = pred / prices[ticker].iloc[-1] - 1
        return forecasts

    ml_forecasts = pd.Series(forecast_returns(prices), name="Predicted Returns")
    
    with st.expander("Statistical Return Predictions"):
        fig_ml = px.bar(ml_forecasts, labels={'value': 'Predicted Return', 'index': 'Asset'})
        st.plotly_chart(fig_ml)

    # ======================
    # Black-Litterman Model
    # ======================
    market_caps = np.ones(num_assets)/num_assets  # Simplified market prior
    pi = delta * S @ market_caps
    
    try:
        P_mat = np.array(P)
        Q_vec = np.array(Q)
        omega = np.diag([(1/c - 1)*p@S@p.T for c,p in zip(confidences, P_mat)])
        
        inv_tau_S = np.linalg.inv(tau * S)
        M = np.linalg.inv(inv_tau_S + P_mat.T @ np.linalg.inv(omega) @ P_mat)
        bl_returns = M @ (inv_tau_S @ pi + P_mat.T @ np.linalg.inv(omega) @ Q_vec)
        bl_cov = S + M
    except np.linalg.LinAlgError:
        st.error("Matrix inversion failed - check view consistency")
        st.stop()

    # Blend ML forecasts with BL returns
    blend_ratio = st.slider("ML Forecast Influence", 0.0, 1.0, 0.3)
    combined_returns = (1 - blend_ratio) * bl_returns + blend_ratio * ml_forecasts.values

    # ... [Rest of Black-Litterman code remains same but use combined_returns] ...

    # ======================

    # ======================
    # Portfolio Optimization
    # ======================
    weights = cp.Variable(num_assets)
    ret = bl_returns @ weights
    risk = cp.quad_form(weights, bl_cov)
    transaction_cost = lambda_tc * cp.norm(weights, 1)
    
    problem = cp.Problem(
        cp.Maximize(ret - delta*risk - transaction_cost),
        [
            cp.sum(weights) == 1,
            weights >= min_weight,
            weights <= max_weight
        ]
    )
    problem.solve()
    
    if weights.value is None:
        st.error("Optimization failed - relax constraints")
        st.stop()

    # ======================
    # Results Visualization
    # ======================
    st.header("📈 Optimal Portfolio")
    optimal_weights = pd.Series(weights.value.round(3), index=tickers)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        fig = px.pie(optimal_weights, names=optimal_weights.index,
                    values=optimal_weights.values, hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        portfolio_ret = optimal_weights @ bl_returns
        portfolio_vol = np.sqrt(optimal_weights @ bl_cov @ optimal_weights)
        sharpe = portfolio_ret / portfolio_vol
        
        st.metric("Expected Return", f"{portfolio_ret*100:.2f}%", 
                 help="Annualized excess return over risk-free rate")
        st.metric("Expected Volatility", f"{portfolio_vol*100:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        st.write("**Weight Constraints**")
        st.write(f"Min: {min_weight*100}% | Max: {max_weight*100}%")
        st.write(f"Transaction Cost Impact: {lambda_tc*100:.1f}% of trades")

    # ======================
    # Risk Analysis
    # ======================
    st.header("🛡️ Risk Breakdown")
    total_var = optimal_weights @ bl_cov @ optimal_weights
    marginal_risk = bl_cov @ optimal_weights
    risk_contrib = np.multiply(optimal_weights, marginal_risk) / total_var
    
    fig2 = px.bar(x=risk_contrib*100, y=tickers, orientation='h',
                 labels={'x':'Risk Contribution (%)', 'y':'Asset'})
    st.plotly_chart(fig2, use_container_width=True)

    # ======================
    # Sensitivity Analysis
    # ======================
    st.header("📉 Parameter Sensitivity")
    deltas = np.linspace(0.01, 1, 20)
    sensitivity = []
    for d in deltas:
        problem = cp.Problem(
            cp.Maximize(bl_returns @ weights - d * cp.quad_form(weights, bl_cov)),
            [cp.sum(weights) == 1, weights >= min_weight, weights <= max_weight]
        )
        problem.solve()
        if weights.value is not None:
            sensitivity.append(weights.value)
    
    sensitivity_df = pd.DataFrame(sensitivity, 
                                 index=deltas.round(2), 
                                 columns=tickers)
    fig3 = px.line(sensitivity_df, labels={'index':'Risk Aversion', 'value':'Weight'})
    st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()
