import numpy as np
import scipy.stats as ss

class GeometricBrownianMotion:
    """Class for generating geometric Brownian motion paths."""

    def __init__(self, S0: float, mu: float, sigma: float, N: int, T: float, M: int):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.N = N
        self.T = T
        self.M = M

    def generate_paths(self):
        """Generate geometric Brownian motion paths."""
        dt = self.T / self.N
        t = np.linspace(0, self.T, self.N)
        dW = ss.norm.rvs(scale=np.sqrt(dt), size=(self.N - 1, self.M))
        W = np.cumsum(dW, axis=0)
        W = np.vstack([np.zeros(self.M), W])
        S = self.S0 * np.exp((self.mu - 0.5 * self.sigma**2) * t[:, None] + self.sigma * W)
        return t, S

