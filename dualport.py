import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.risk_models import CovarianceShrinkage
from scipy.optimize import minimize
from matplotlib.figure import Figure
from datetime import datetime, timedelta

# Streamlit layout
st.title('Portfolio Optimization')

# User input for tickers
tickers = st.text_input('Enter your stock tickers separated by comma:').split(',')

# Removing leading and trailing white spaces in each ticker
tickers = [ticker.strip() for ticker in tickers]

# Get today's date
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Get data
def download_data(tickers, start_date, end_date):
    price_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return price_data

data = download_data(tickers, start_date, end_date)
data.dropna(inplace=True)

# Calculate returns and covariance
mu = expected_returns.mean_historical_return(data)
S = CovarianceShrinkage(data).ledoit_wolf()

# Efficient Frontier
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
ef.portfolio_performance(verbose=True)

# Risk Parity
def risk_parity(S, w_guess, risk_tol):
    def objective(w, S, risk_tol):
        variance = w @ S @ w
        sigma = np.sqrt(variance)
        mrc = 1/sigma * (S @ w)
        rc = w * mrc
        risk_diffs = rc[:, np.newaxis] - rc
        sum_squared_risk_diffs = np.sum(np.square(np.abs(risk_diffs)))
        return sum_squared_risk_diffs

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0,1) for x in range(len(w_guess)))
    result = minimize(objective, w_guess, args=(S, risk_tol), method='SLSQP', constraints=constraints, bounds=bounds)
    return result.x

n = len(tickers)
w_guess = np.ones(n) / n
risk_tol = np.ones(n) / n
risk_parity_weights = risk_parity(S.to_numpy(), w_guess, risk_tol)
risk_parity_weights = {tickers[i]: risk_parity_weights[i] for i in range(n)}

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10,10))

axs[0, 0].pie(cleaned_weights.values(), labels=cleaned_weights.keys(), autopct='%1.1f%%')
axs[0, 0].set_title('Efficient Frontier Portfolio Composition')

axs[0, 1].plot((1 + (data.pct_change() * list(cleaned_weights.values())).sum(axis=1)).cumprod())
axs[0, 1].set_title('Efficient Frontier Portfolio Returns')

axs[1, 0].pie(risk_parity_weights.values(), labels=risk_parity_weights.keys(), autopct='%1.1f%%')
axs[1, 0].set_title('Risk Parity Portfolio Composition')

axs[1, 1].plot((1 + (data.pct_change() * list(risk_parity_weights.values())).sum(axis=1)).cumprod())
axs[1, 1].set_title('Risk Parity Portfolio Returns')

plt.tight_layout()
st.pyplot(fig)
