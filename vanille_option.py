import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yfinance as yf
import logging
import scipy.stats as ss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def knock_out_up_option(S0, K, T, sigma, r, q, H, N, M):
    """
    Knock-out option pricing using Monte Carlo simulation.
    """
    dt = T / N
    t = np.linspace(0, T, N)
    dW = ss.norm.rvs(scale=np.sqrt(dt), size=(N - 1, M))
    W = np.cumsum(dW, axis=0)
    W = np.vstack([np.zeros(M), W])
    S = S0 * np.exp((r - q - 0.5 * sigma**2) * t[:, None] + sigma * W)
    S = np.maximum(S, H)
    plt.plot(S)
    plt.show()
    payoff = np.maximum(S - K, 0)
    return np.mean(payoff)

if __name__ == "__main__":
    pass