import numpy as np
import scipy.stats as ss

class GirsanovSimulator:
    """Class for simulating paths under risk-neutral measure using Girsanov theorem."""

    def __init__(self, S0: float, mu: float, r: float, sigma: float, N: int, T: float, M: int):
        self.S0 = S0
        self.mu = mu
        self.r = r
        self.sigma = sigma
        self.N = N
        self.T = T
        self.M = M

    def generate_paths(self):
        """Generate paths under risk-neutral measure."""
        dt = self.T / self.N
        t = np.linspace(0, self.T, self.N)

        theta = (self.mu - self.r) / self.sigma
        dW = ss.norm.rvs(scale=np.sqrt(dt), size=(self.N - 1, self.M))
        dW_tilde = dW - theta * np.sqrt(dt)

        W_tilde = np.cumsum(dW_tilde, axis=0)
        W_tilde = np.vstack([np.zeros(self.M), W_tilde])

        return self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * t[:, None] + self.sigma * W_tilde)

