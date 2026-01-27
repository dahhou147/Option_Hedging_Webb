import numpy as np
import scipy.stats as ss
from pricing_model import BlackScholesPricer

class Greeks:
    """Class for Greeks calculations."""

    def __init__(self, pricer: BlackScholesPricer):
        self.pricer = pricer.copy()

    def delta(self, S, tau, option_type="call"):
        """Calculate delta of the option."""
        d1, _ = self.pricer._d1_d2(S, tau)
        if option_type == "call":
            return np.exp(-self.pricer.q * tau) * ss.norm.cdf(d1)
        if option_type == "put":
            return -np.exp(-self.pricer.q * tau) * ss.norm.cdf(-d1)
        raise ValueError("option_type must be 'call' or 'put'")

    def gamma(self, S, tau):
        """Calculate gamma of the option."""
        d1, _ = self.pricer._d1_d2(S, tau)
        return (
            np.exp(-self.pricer.q * tau) * ss.norm.pdf(d1) / (S * self.pricer.sigma * np.sqrt(tau))
        )

    def vega(self, S, tau):
        """Calculate vega of the option."""
        d1, _ = self.pricer._d1_d2(S, tau)
        return S * np.exp(-self.pricer.q * tau) * ss.norm.pdf(d1) * np.sqrt(tau)

    def theta(self, S, tau, option_type="call"):
        """Calculate theta of the option."""
        d1, d2 = self.pricer._d1_d2(S, tau)
        if option_type == "call":
            return -S * self.pricer.sigma * np.exp(-self.pricer.q * tau) * ss.norm.pdf(d1) / (
                2 * np.sqrt(tau)
            ) - self.pricer.r * self.pricer.K * np.exp(-self.pricer.r * tau) * ss.norm.cdf(d2)
        if option_type == "put":
            return -S * self.pricer.sigma * np.exp(-self.pricer.q * tau) * ss.norm.pdf(d1) / (
                2 * np.sqrt(tau)
            ) + self.pricer.r * self.pricer.K * np.exp(-self.pricer.r * tau) * ss.norm.cdf(-d2)
        raise ValueError("option_type must be 'call' or 'put'")

if __name__ == "__main__":
    pricer = BlackScholesPricer(100, 100, 1, 0.2, 0.05, 0.0)
    greeks = Greeks(pricer)
    print(greeks.delta(100, 1))
    print(greeks.gamma(100, 1))
    print(greeks.vega(100, 1))
    print(greeks.theta(100, 1))