import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
from .black_scholes import BlackScholesPricer

class VolatilitySmile:
    """Class for volatility smile calculations."""

    def __init__(self, pricer: BlackScholesPricer):
        self.pricer = pricer

    def implied_volatility(self, strike: float, market_price: float):
        """Calculate implied volatility using Newton-Raphson method."""

        def f(sigma): 
            temp_pricer = self.pricer.copy()  # éviter les effets de bord
            temp_pricer.sigma = sigma
            temp_pricer.K = strike
            return temp_pricer.price_call() - market_price

        try:
            return so.newton(f, 0.2, maxiter=50)
        except RuntimeError:
            return np.nan

    def volatility_smile(self, strikes: np.ndarray, market_prices: np.ndarray):
        """Calculate the implied volatility curve."""
        return np.array(
            [
                self.implied_volatility(strike, price)
                for strike, price in zip(strikes, market_prices)
            ]
        )

    def plot_smile(self, strikes, market_prices):
        """Plot the implied volatility curve."""
        smile = self.volatility_smile(strikes, market_prices)
        plt.figure(figsize=(10, 6))
        plt.plot(strikes, smile * 100, "o-")
        plt.xlabel("strike price")
        plt.ylabel("implied volatility (%)")
        plt.title("Volatility Smile")
        plt.grid(True)
        return plt.gcf()

