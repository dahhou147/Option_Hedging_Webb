# %%
from calibration.calibration import main
from pricer.pricing_model import ConstructPortfolio, BlackScholesPricer, GirsanovSimulator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

N = 252
M = 100


class Launcher:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.df = main(ticker)
        self.maturities = self.df["expiry_date"].unique()
        self.expiry_date = self.maturities[0]
        self.K = self.df["strike"].iloc[0]
        self.K1 = self.df["strike"].iloc[1]
        self.K2 = self.df["strike"].iloc[2]
        self.spot = self.df["spot_price"].iloc[0]

    def launch(self):
        df = self.df[self.df["expiry_date"] == self.expiry_date]
        pricer = BlackScholesPricer(
            self.spot,
            self.K,
            self.df["maturity"].iloc[0],
            self.df["implied_volatility"].iloc[0],
            self.df["risk_free_rate"].iloc[0],
            self.df["dividend_yield"].iloc[0],
        )
        simulator = GirsanovSimulator(
            self.spot,
            self.df["drift"].iloc[0],
            self.df["risk_free_rate"].iloc[0],
            self.df["implied_volatility"].iloc[0],
            N,
            self.df["maturity"].iloc[0],
            M,
        )
        paths = simulator.generate_paths()
        options_types = ["call", "put", "call"]
        implied_volatility = [
            df["implied_volatility"].iloc[0],
            df["implied_volatility"].iloc[1],
            df["implied_volatility"].iloc[2],
        ]
        portfolio = ConstructPortfolio(
            pricer, paths, self.K1, self.K2, options_types, implied_volatility
        )
        logger.info("launching portfolio hedger ")
        try:
            portfolio.hedge_portfolio()
        except Exception as e:
            logger.error(f"Error during portfolio hedging: {e}")
            return None
        logger.info("done")
        return portfolio

    def run(self):
        portfolio = self.launch()
        if portfolio is None:
            print("Portfolio launch failed. Exiting run.")
            return
        final_pnl = portfolio.pnl[-1, :]
        mean_pnl = np.mean(final_pnl)
        std_pnl = np.std(final_pnl)
        daily_returns = np.diff(portfolio.portfolio_values, axis=0) / portfolio.portfolio_values[:-1]
        cumulative_returns = np.cumprod(1 + daily_returns, axis=0)
        running_max = np.maximum.accumulate(cumulative_returns, axis=0)
        drawdowns = (running_max - cumulative_returns) / running_max

        print("\nHedging Strategy Performance Metrics:")
        print(f"Mean PnL: ${mean_pnl:.2f}")
        print(f"PnL Standard Deviation: ${std_pnl:.2f}")
        print(f"Maximum Drawdown: {np.max(drawdowns)*100:.2f}%")
        positive_returns = np.sum(daily_returns > 0) / daily_returns.size
        print(f"Win Rate: {positive_returns*100:.2f}%")

        # Calculate Value at Risk (VaR)
        var_95 = np.percentile(daily_returns, 5)
        print(f"95% Value at Risk: {var_95*100:.2f}%")

        # Create visualization
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))

        # Plot mean daily returns over time
        axes[0, 0].plot(np.mean(daily_returns, axis=1))
        axes[0, 0].set_title('Mean Daily Returns Over Time')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Mean Return')

        # Plot cumulative PnL
        pnl_mean = np.mean(portfolio.pnl, axis=1)
        cumulative_pnl = np.cumsum(portfolio.pnl, axis=0)
        mean_cumulative_pnl = np.mean(cumulative_pnl, axis=1)
        std_cumulative_pnl = np.std(cumulative_pnl, axis=1)
        time_points = np.linspace(0, 1, N)

        axes[0, 1].plot(time_points, mean_cumulative_pnl, label='Mean PnL', color="red")
        axes[0, 1].fill_between(time_points,
                                mean_cumulative_pnl - std_cumulative_pnl,
                                mean_cumulative_pnl + std_cumulative_pnl,
                                alpha=0.2)
        axes[0, 1].set_title('Cumulative PnL')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Cumulative PnL')
        axes[0, 1].legend()

        # Mean PnL
        axes[1, 0].plot(pnl_mean)
        axes[1, 0].set_title('Average PnL')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('PnL')

        # Average Drawdown
        mean_drawdowns = np.mean(drawdowns, axis=1)
        axes[1, 1].plot(time_points[:-1], mean_drawdowns)
        axes[1, 1].set_title('Average Drawdown')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Drawdown')

        # Average Hedge Ratios
        mean_hedge_ratios = np.mean(portfolio.old_coeffs, axis=1)
        axes[2, 0].plot(time_points, mean_hedge_ratios[:, 0], label='Option 1')
        axes[2, 0].plot(time_points, mean_hedge_ratios[:, 1], label='Option 2')
        axes[2, 0].plot(time_points, mean_hedge_ratios[:, 2], label='Stock')
        axes[2, 0].set_title('Average Hedge Ratios')
        axes[2, 0].set_xlabel('Time Step')
        axes[2, 0].set_ylabel('Hedge Ratio')
        axes[2, 0].legend()

        # Simulated Paths
        axes[2, 1].plot(portfolio.paths)
        axes[2, 1].set_title('Simulated Paths')
        axes[2, 1].set_xlabel('Time Step')
        axes[2, 1].set_ylabel('Price')
        
        # distribution of underline asset
        sns.kdeplot(portfolio.paths[-1, :], ax=axes[3, 0], fill=True)
        axes[3, 0].axvline(self.K, color='red', linestyle='--', label='Strike Price')
        axes[3, 0].axvline(self.K1, color='blue', linestyle='--', label='Strike Price 1')
        axes[3, 0].axvline(self.K2, color='green', linestyle='--', label='Strike Price 2')
        axes[3, 0].axvline(self.spot, color='black', linestyle='--', label='Spot Price')
        axes[3, 0].set_title('Distribution of Underlying Asset')
        axes[3, 0].set_xlabel('Price')
        axes[3, 0].set_ylabel('Density')
        axes[3, 0].legend()
        
        
        plt.suptitle(f'Hedging Strategy Analysis for {self.ticker}', fontsize=16)
        plt.tight_layout()
        plt.show()

#%%
if __name__ == "__main__":
    ticker = "MSFT"
    launcher = Launcher(ticker)
    portfolio = launcher.launch()
    launcher.run()
# %%


nums = [1, 2, 3, 4, 5]
s = max(nums)
nums.index(s)  # Returns the index of the maximum value in the list