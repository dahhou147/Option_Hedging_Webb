import numpy as np
import scipy.stats as ss

class BlackScholesPricer:
    """Class for Black-Scholes option pricing."""

    def __init__(self, S0: float, K: float, T: float, sigma: float, r: float, q: float = 0.0):
        self.S0 = S0
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r
        self.q = q

    def _d1_d2(self, S, tau):
        """Compute d1 and d2 for Black-Scholes formula."""
        d1 = (np.log(S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * tau) / (
            self.sigma * np.sqrt(tau)
        )
        d2 = d1 - self.sigma * np.sqrt(tau)
        return d1, d2

    def price_call(self):
        """Black-Scholes call price."""
        d1, d2 = self._d1_d2(self.S0, self.T)
        return self.S0 * np.exp(-self.q * self.T) * ss.norm.cdf(d1) - self.K * np.exp(
            -self.r * self.T
        ) * ss.norm.cdf(d2)

    def price_put(self):
        """Black-Scholes put price."""
        d1, d2 = self._d1_d2(self.S0, self.T)
        return self.K * np.exp(-self.r * self.T) * ss.norm.cdf(-d2) - self.S0 * np.exp(
            -self.q * self.T
        ) * ss.norm.cdf(-d1)


    def copy(self):
        return BlackScholesPricer(self.S0, self.K, self.T, self.sigma, self.r, self.q)

