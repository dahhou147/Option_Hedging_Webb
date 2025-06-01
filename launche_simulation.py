# %%
from calibration import main
from pricing_model import ConstructPortfolio, BlackScholesPricer, GirsanovSimulator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

N = 252
M = 1000


class Launcher:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.df = main(ticker)
        self.maturities = self.df["expiry_date"].unique()
        self.expiry_date = self.maturities[0]

    def launch(self):
        df = self.df[self.df["expiry_date"] == self.expiry_date]
        pricer = BlackScholesPricer(
            self.df["spot_price"].iloc[0],
            self.df["strike"].iloc[0],
            self.df["maturity"].iloc[0],
            self.df["implied_volatility"].iloc[0],
            self.df["risk_free_rate"].iloc[0],
            self.df["dividend_yield"].iloc[0],
        )
        simulator = GirsanovSimulator(
            self.df["spot_price"].iloc[0],
            self.df["drift"].iloc[0],
            self.df["risk_free_rate"].iloc[0],
            self.df["implied_volatility"].iloc[0],
            N,
            self.df["maturity"].iloc[0],
            M,
        )
        paths = simulator.generate_paths()
        K1 = df["strike"].iloc[1]
        K2 = df["strike"].iloc[2]
        options_types = ["call", "put", "call"]
        implied_volatility = [df["implied_volatility"].iloc[0], df["implied_volatility"].iloc[1], df["implied_volatility"].iloc[2]]
        portfolio = ConstructPortfolio(
            pricer, paths, K1, K2, options_types, implied_volatility
        )
        print("launching portfolio hedger ")
        portfolio.hedge_portfolio()
        print("done")
        return portfolio
        

#%%
if __name__ == "__main__":
    # Example usage
    ticker = "MSFT"
    launcher = Launcher(ticker)
    portfolio = launcher.launch()
    final_pnl = portfolio.pnl[-1, :]
    # mean_pnl = np.mean(final_pnl)   
    # std_pnl = np.std(final_pnl)
    # sharpe_ratio = np.sqrt(252) * mean_pnl / std_pnl if std_pnl != 0 else 0
    
    # daily_returns = np.diff(portfolio.portfolio_values, axis=0) / portfolio.portfolio_values[:-1]
    
    # cumulative_returns = np.cumprod(1 + daily_returns, axis=0)
    # running_max = np.maximum.accumulate(cumulative_returns, axis=0)
    # drawdowns = (running_max - cumulative_returns) / running_max
    
    # print("\nHedging Strategy Performance Metrics:")
    # print(f"Mean PnL: ${mean_pnl:.2f}")
    # print(f"PnL Standard Deviation: ${std_pnl:.2f}")
    # print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    # print(f"Maximum Drawdown: {np.max(drawdowns)*100:.2f}%")
    
    # # Calculate additional metrics
    # positive_returns = np.sum(daily_returns > 0) / daily_returns.size
    # print(f"Win Rate: {positive_returns*100:.2f}%")
    
    # # Calculate Value at Risk (VaR)
    # var_95 = np.percentile(daily_returns, 5)
    # print(f"95% Value at Risk: {var_95*100:.2f}%")
    
    # # Create visualization
    # fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # # Plot PnL distribution
    # sns.histplot(daily_returns.flatten(), ax=axes[0, 0], kde=True)
    # axes[0, 0].set_title('Daily Returns Distribution')
    # axes[0, 0].set_xlabel('Return')
    # axes[0, 0].set_ylabel('Frequency')
    
    # # Plot cumulative PnL
    # cumulative_pnl = np.cumsum(portfolio.pnl, axis=0)
    # mean_cumulative_pnl = np.mean(cumulative_pnl, axis=1)
    # std_cumulative_pnl = np.std(cumulative_pnl, axis=1)
    # time_points = np.arange(N)
    
    # axes[0, 1].plot(time_points, mean_cumulative_pnl, label='Mean PnL')
    # axes[0, 1].fill_between(time_points, 
    #                         mean_cumulative_pnl - std_cumulative_pnl,
    #                         mean_cumulative_pnl + std_cumulative_pnl,
    #                         alpha=0.2)
    # axes[0, 1].set_title('Cumulative PnL')
    # axes[0, 1].set_xlabel('Time Step')
    # axes[0, 1].set_ylabel('Cumulative PnL')
    # axes[0, 1].legend()
    
    # mean_drawdowns = np.mean(drawdowns, axis=1)
    # axes[1, 0].plot(time_points[:-1], mean_drawdowns)
    # axes[1, 0].set_title('Average Drawdown')
    # axes[1, 0].set_xlabel('Time Step')
    # axes[1, 0].set_ylabel('Drawdown')
    
    # mean_hedge_ratios = np.mean(portfolio.old_coeffs, axis=1)
    # axes[1, 1].plot(time_points, mean_hedge_ratios[:, 0], label='Option 1')
    # axes[1, 1].plot(time_points, mean_hedge_ratios[:, 1], label='Option 2')
    # axes[1, 1].plot(time_points, mean_hedge_ratios[:, 2], label='Stock')
    # axes[1, 1].set_title('Average Hedge Ratios')
    # axes[1, 1].set_xlabel('Time Step')
    # axes[1, 1].set_ylabel('Hedge Ratio')
    # axes[1, 1].legend()
    
    # plt.suptitle(f'Hedging Strategy Analysis for {ticker}', fontsize=16)
    # plt.tight_layout()
    # plt.show()

# %%
