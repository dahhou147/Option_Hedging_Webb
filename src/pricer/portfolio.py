import numpy as np
from .black_scholes import BlackScholesPricer
from .greeks import Greeks

class ConstructPortfolio:
    """Class for constructing a self-financing hedging portfolio."""

    def __init__(
        self,
        pricer: BlackScholesPricer,
        paths: np.ndarray,
        K1: float,
        K2: float,
        options_types: list[str],
        implied_volatility: list[float],
    ):
        self.pricer = pricer
        self.paths = paths
        self.N = paths.shape[0]
        self.T = pricer.T
        self.K1 = K1
        self.K2 = K2
        self.K = self.pricer.K
        self.dt = self.T / self.N
        self.M = paths.shape[1]
        self.options_types = options_types
        self.implied_volatility = implied_volatility
        self.pnl = np.zeros((self.N, self.M))
        self.portfolio_values = np.zeros((self.N, self.M))
        self.cash_positions = np.zeros((self.N, self.M))
        self.old_coeffs = np.zeros((self.N, self.M, 3))
        self.options_prices = np.zeros((self.N, self.M))

    def _get_pricer(self, S, K, tau, implied_volatility):
        pricer = self.pricer.copy()
        pricer.S0 = S
        pricer.K = K
        pricer.T = tau
        pricer.sigma = implied_volatility
        return pricer

    def _get_greeks(self, K, S, tau, option_type, implied_volatility):
        pricer = self._get_pricer(S, K, tau, implied_volatility)
        greeks = Greeks(pricer)
        delta = greeks.delta(S, tau, option_type)
        gamma = greeks.gamma(S, tau)
        vega = greeks.vega(S, tau)
        return delta, gamma, vega

    def _get_option_price(self, K, S, tau, option_type, implied_vol):
        pricer = self._get_pricer(S, K, tau, implied_vol)
        return pricer.price_call() if option_type == "call" else pricer.price_put()

    def _get_options_values(self, S, tau, implied_volatility):
        Ks = [self.K, self.K1, self.K2]
        option_types = self.options_types
        implied_volatilities = implied_volatility
        [val0, val1, val2] = [
            self._get_option_price(K, S, tau, option_type, implied_volatility)
            for K, option_type, implied_volatility in zip(Ks, option_types, implied_volatilities)
        ]
        return val0, val1, val2

    def get_coefficients(self, S, tau, implied_volatility):
        """Solve for portfolio weights:
        alpha1 * greeks(option1) + alpha2 * greeks(option2) + alpha3 * greeks(stock) = target greeks
        """
        delta0, gamma0, vega0 = self._get_greeks(
            self.K, S, tau, self.options_types[0], implied_volatility[0]
        )

        delta1, gamma1, vega1 = self._get_greeks(
            self.K1, S, tau, self.options_types[1], implied_volatility[1]
        )

        delta2, gamma2, vega2 = self._get_greeks(
            self.K2, S, tau, self.options_types[2], implied_volatility[2]
        )

        matrix = np.array(
            [
                [delta1, delta2, 1],
                [gamma1, gamma2, 0],
                [vega1, vega2, 0],
            ]
        )
        b = np.array([delta0, gamma0, vega0])

        cond = np.linalg.cond(matrix)

        if cond > 1e10:
            lambda_reg = 1e-6
            matrix_reg = matrix.T @ matrix + lambda_reg * np.eye(3)
            coeffs = np.linalg.solve(matrix_reg, matrix.T @ b)
        else:
            try:
                coeffs = np.linalg.solve(matrix, b)
            except np.linalg.LinAlgError as e:
                print(e)
                return None
        return coeffs


    def hedge_portfolio(self):
        """Perform dynamic hedging of the portfolio"""
        for i in range(self.M):
            S = self.paths[0, i]
            tau = self.T
            coeffs = self.get_coefficients(S, tau, self.implied_volatility)

            if coeffs is None:
                continue
                
            val0, val1, val2 = self._get_options_values(S, tau, self.implied_volatility)
            cash = val0 - (coeffs[0] * val1 + coeffs[1] * val2 + coeffs[2] * S)
            self.options_prices[0, i] = val0
            self.cash_positions[0, i] = cash
            self.portfolio_values[0, i] = coeffs[0] * val1 + coeffs[1] * val2 + coeffs[2] * S + cash
            self.pnl[0, i] = self.portfolio_values[0, i] - val0
            self.old_coeffs[0, i, :] = coeffs

            old_coeffs = coeffs

            for j in range(1, self.N):
                S = self.paths[j, i]
                tau = self.T - j * self.dt
                coeffs = self.get_coefficients(S, tau, self.implied_volatility)
                if coeffs is None:
                    coeffs = old_coeffs

                val0, val1, val2 = self._get_options_values(S, tau, self.implied_volatility)
                self.options_prices[j, i] = val0
                cash = self.cash_positions[j - 1, i] * np.exp(self.pricer.r * self.dt)

                d_alpha = coeffs - old_coeffs
                cash -= d_alpha[0] * val1 + d_alpha[1] * val2 + d_alpha[2] * S

                self.cash_positions[j, i] = cash
                self.portfolio_values[j, i] = (
                    coeffs[0] * val1 + coeffs[1] * val2 + coeffs[2] * S + cash
                )
                self.pnl[j, i] = self.portfolio_values[j, i] - val0
                self.old_coeffs[j, i, :] = coeffs

                old_coeffs = coeffs

