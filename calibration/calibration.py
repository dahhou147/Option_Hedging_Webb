# %%
from datetime import datetime
from typing import Optional, Tuple, Dict
import numpy as np
import yfinance as yf
import pandas as pd
import logging
from pricer.pricing_model import BlackScholesPricer, VolatilitySmile

ANNUALIZATION_FACTOR = 252
DEFAULT_RISK_FREE_RATE = 0.03
MIN_TRADING_VOLUME = 10
MIN_IMPLIED_VOL = 0.01
MAX_IMPLIED_VOL = 2.0
logging.basicConfig(
    level=logging.INFO,  # ou DEBUG, WARNING, etc.
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class GetMarketData:
    """
    Class for calibrating option pricing model parameters from market data.
    This class handles the fetching of market data, calculation of implied volatilities,
    and calibration of model parameters for a specific stock ticker.
    """

    def __init__(self, ticker: str):
        """
        Initialize the calibrator with a specific ticker.

        Args:
            ticker (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
        """
        if not isinstance(ticker, str) or not ticker:
            raise ValueError("Ticker must be a non-empty string")

        self.ticker = ticker
        self.stock_data: Optional[pd.DataFrame] = None
        self.option_chain = None
        self.spot_price: Optional[float] = None
        self.mu: Optional[float] = None
        self.risk_free_rate: Optional[float] = None
        self.dividend_yield: Optional[float] = None
        self.historical_volatility: Optional[float] = None
        self.expiry_dates = None

    def fetch_market_data(self, period: str = "1y") -> None:
        """
        Fetch market data for the specified ticker.

        Args:
            period (str): Period for historical data (e.g., '1y', '6mo')

        Raises:
            ValueError: If data fetching fails or required data is missing
        """
        try:
            stock = yf.Ticker(self.ticker)
            self.stock_data = stock.history(period=period)

            if self.stock_data.empty:
                raise ValueError(f"No historical data available for {self.ticker}")

            self.spot_price = float(self.stock_data["Close"].iloc[-1])

            log_returns = np.log(
                self.stock_data["Close"] / self.stock_data["Close"].shift(1)
            )
            self.historical_volatility = float(
                log_returns.std() * np.sqrt(ANNUALIZATION_FACTOR)
            )
            self.mu = float(log_returns.mean() * ANNUALIZATION_FACTOR)

            self.dividend_yield = float(stock.info.get("dividendYield", 0.0))
            self.risk_free_rate = DEFAULT_RISK_FREE_RATE
            self.option_chain = stock.option_chain
            self.expiry_dates = stock.options

        except Exception as e:
            raise ValueError(f"Failed to fetch market data: {str(e)}")

    def get_option_data(self, expiry_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retrieve option data for a specific expiry date.

        Args:
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format

        Returns:
            tuple: (calls_df, puts_df) DataFrames for calls and puts

        Raises:
            ValueError: If option data is not available
        """
        if self.option_chain is None:
            raise ValueError(
                "Option data not available. Run fetch_market_data() first."
            )

        try:
            options = self.option_chain(expiry_date)
            return options.calls, options.puts
        except Exception as e:
            logger.error(f"No options available for date {expiry_date}: {e}")
            return None, None

    def calculate_time_to_maturity(self, expiry_date: str) -> float:
        """
        Calculate time to maturity in years.

        Args:
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format

        Returns:
            float: Time to maturity in years
        """
        try:
            expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
            today = datetime.now()
            days = (expiry - today).days
            return max(days, 1) / 365.0
        except ValueError as e:
            logger.error(f"Invalid date format. Use YYYY-MM-DD: {str(e)}")
            return 0

    def calibrate_implied_volatility(
        self, expiry_date: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Calibrate implied volatility for different strikes and return data in memory.

        Args:
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format

        Returns:
            tuple: (strikes, implied_vols, volatility_data) Strike prices, implied volatilities and data dictionary
        """
        calls, _ = self.get_option_data(expiry_date)
        if calls is None:
            logger.error("No call options data available")

        calls = calls[calls["volume"] > MIN_TRADING_VOLUME]
        if calls.empty:
            logger.error("No options with sufficient trading volume")

        maturity = self.calculate_time_to_maturity(expiry_date)

        pricer = BlackScholesPricer(
            S0=self.spot_price,
            K=100,  # he will be changed
            T=maturity,
            sigma=self.historical_volatility,
            r=self.risk_free_rate,
            q=self.dividend_yield,
        )
        smile_calculator = VolatilitySmile(pricer)

        volatility_data = {
            "ticker": self.ticker,
            "maturity": maturity,
            "expiry_date": expiry_date,
            "risk_free_rate": self.risk_free_rate,
            "dividend_yield": self.dividend_yield,
            "drift": self.mu,
            "spot_price": self.spot_price,
            "strikes": {},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        strikes = []
        implied_vols = []
        i = 0
        for _, row in calls.iterrows():
            strike = row["strike"]
            market_price = row["lastPrice"]

            iv = smile_calculator.implied_volatility(strike, market_price)
            if not np.isnan(iv) and MIN_IMPLIED_VOL < iv < MAX_IMPLIED_VOL:
                strikes.append(strike)
                implied_vols.append(iv)

                volatility_data["strikes"][i] = {
                    "implied_volatility": float(iv),
                    "market_price": float(market_price),
                    "strike": float(strike),
                    "moneyness": float(strike / self.spot_price),
                }
                i += 1

        return np.array(strikes), np.array(implied_vols), volatility_data

    def get_volatility_dataframe(self, volatility_data: Dict) -> pd.DataFrame:
        try:
            df = pd.DataFrame.from_dict(volatility_data["strikes"], orient="index")
            df["risk_free_rate"] = volatility_data["risk_free_rate"]
            df["dividend_yield"] = volatility_data["dividend_yield"]
            df["ticker"] = volatility_data["ticker"]
            df["expiry_date"] = volatility_data["expiry_date"]
            df["spot_price"] = volatility_data["spot_price"]
            df["maturity"] = volatility_data["maturity"]
            df["drift"] = volatility_data["drift"]
            df["timestamp"] = volatility_data["timestamp"]

            df = df.sort_values("strike")

            return df

        except Exception as e:
            logger.error(f"Error converting volatility data: {str(e)}")
            return pd.DataFrame()

    def get_all_maturities_data(self) -> pd.DataFrame:
        all_data = []

        for expiry in self.expiry_dates:
            try:
                _, _, volatility_data = self.calibrate_implied_volatility(expiry)
                df = self.get_volatility_dataframe(volatility_data)
                all_data.append(df)
            except Exception as e:
                logger.error(f"Error processing expiry {expiry}: {str(e)}")
                continue

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)


def main(ticker: str) -> pd.DataFrame:
    calibrator = GetMarketData(ticker)
    calibrator.fetch_market_data()
    return calibrator.get_all_maturities_data()


# %%
if __name__ == "__main__":
    ticker = "AAPL"
    df = main(ticker)
    logger.info(df)
# %%
