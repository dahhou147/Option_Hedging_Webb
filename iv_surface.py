#%%
import scipy.stats as ss
import scipy.optimize as so
from scipy import interpolate
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay
import pandas as pd
import numpy as np
from pricing_model import BlackScholesPricer
from openbb import obb
from config.logging_config import setup_logger

logger = setup_logger(name="iv_surface", log_file=None)

class DataManager:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = obb.equity.price.historical(symbol=self.ticker).to_df()
        self.options = obb.derivatives.options.chains(symbol=self.ticker).to_df()
        self.free_rate = self.get_free_rate()
        self.spot_price = self.data["close"].iloc[-1]
    
    def get_free_rate(self):
        df = obb.economy.interest_rates(country="USA", maturity="10Y").to_df()
        if not df.empty:
            return df["value"].iloc[-1] 
        else:
            logger.warning("10-year US Treasury rate not found, defaulting to 0.03")
            return 0.03




class IvBlackScholes(DataManager):
    def __init__(self, ticker):
        super().__init__(ticker=ticker)
        self.processed_options = self.preprocessing()
        self.market_data = self.apply_kernel()
        self.interpolated_surface = None
        self.cordonates = {}

    def preprocessing(self):
        df = self.options.copy()
        df["time_to_expiry"] = (
            pd.to_datetime(df["expiration"])
            - pd.to_datetime(df.get("as_of_date", pd.Timestamp.today()))
        ).dt.days / 365
        df_filtered = df[df["time_to_expiry"] > 0.25].reset_index(drop=True)
        price_col = (
            "last_trade_price"
            if "last_trade_price" in df_filtered.columns
            else ("lastPrice" if "lastPrice" in df_filtered.columns else None)
        )
        if price_col is None:
            raise KeyError(
                "Prix marché introuvable (pas de 'last_trade_price' ni 'lastPrice')"
            )
        df_option = df_filtered.groupby(
            ["strike", "time_to_expiry"], as_index=False
        ).first()[
            ["strike", "time_to_expiry", "option_type", price_col, "implied_volatility"]
        ]
        df_option = df_option.rename(columns={price_col: "market_price"})
        return df_option

    def get_implied_vol(self, strike, time_to_expiry, market_price, option_type):
        def objective_function(volatility):
            pricer = BlackScholesPricer(
                S0=self.spot_price,
                K=strike,
                T=time_to_expiry,
                sigma=volatility,
                r=self.free_rate,
            )
            model_price = (
                pricer.price_call() if option_type == "call" else pricer.price_put()
            )
            return model_price - market_price

        vol_min, vol_max = 0.001, 3.0
        try:
            f_min = objective_function(vol_min)
            f_max = objective_function(vol_max)

            if f_min * f_max <= 0:
                result = so.root_scalar(
                    objective_function,
                    bracket=[vol_min, vol_max],
                    method="brentq",
                    xtol=1e-8,
                )
                return result.root
            else:
                result = so.minimize_scalar(
                    lambda x: objective_function(x) ** 2,
                    bounds=(vol_min, vol_max),
                    method="bounded",
                )
                return result.x

        except:
            return np.nan

    def apply_kernel(self):
        df = self.processed_options.copy()
        df["implied_vol"] = df.apply(
            lambda row: self.get_implied_vol(
                row["strike"],
                row["time_to_expiry"],
                row["market_price"],
                row["option_type"],
            ),
            axis=1,
        )
        return df.assign(spot_price=self.spot_price).assign(
            moyeness=lambda x: x["strike"] / x["spot_price"]
        )

    def interpolate_surface_multimethod(self, grid_size=50):
        """
        Interpole la surface de volatilité en utilisant plusieurs méthodes
        et choisit la meilleure
        """
        if self.market_data is None or len(self.market_data) < 10:
            logger.warning("There are not enough data points for interpolation")
            return None

        df = self.market_data.copy()

        x_min, x_max = df["strike"].min(), df["strike"].max()
        y_min, y_max = df["time_to_expiry"].min(), df["time_to_expiry"].max()

        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1

        x_grid = np.linspace(max(0, x_min - x_padding), x_max + x_padding, grid_size)
        y_grid = np.linspace(max(0.01, y_min - y_padding), y_max + y_padding, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)

        points = df[["strike", "time_to_expiry"]].values
        values = df["implied_vol"].values
        logger.info("Interpolation RBF...")
        try:
            rbf = interpolate.RBFInterpolator(
                points, values, kernel="thin_plate_spline"
            )
            grid_points = np.column_stack((X.ravel(), Y.ravel()))
            Z_rbf = rbf(grid_points).reshape(X.shape)
        except:
            Z_rbf = np.full(X.shape, np.nan)

        logger.info("Linear interpolation...   ")
        tri = Delaunay(points)
        interp_linear = interpolate.LinearNDInterpolator(tri, values)
        Z_linear = interp_linear(X, Y)

        logger.info("Spline interpolation...   ")
        try:
            x_unique = np.unique(points[:, 0])
            y_unique = np.unique(points[:, 1])

            if len(x_unique) > 3 and len(y_unique) > 3:
                interp_spline = interpolate.griddata(
                    points, values, (X, Y), method="cubic", fill_value=np.nan
                )
                Z_spline = interp_spline
            else:
                Z_spline = np.full(X.shape, np.nan)
        except:
            Z_spline = np.full(X.shape, np.nan)

        Z_combined = np.copy(Z_linear)

        nan_mask = np.isnan(Z_combined)
        if not np.all(np.isnan(Z_rbf)):
            Z_combined[nan_mask] = Z_rbf[nan_mask]

        nan_mask = np.isnan(Z_combined)
        if not np.all(np.isnan(Z_spline)):
            Z_combined[nan_mask] = Z_spline[nan_mask]


        Z_smoothed = gaussian_filter(Z_combined, sigma=1.0)

        Z_smoothed = np.clip(Z_smoothed, 0.01, 1.5)

        self.interpolated_surface = {
            "strikes_grid": x_grid,
            "maturities_grid": y_grid,
            "strike_grid_2d": X,
            "maturity_grid_2d": Y,
            "volatility_grid": Z_smoothed,
            "original_points": points,
            "original_values": values,
        }

        logger.info("Interpolation completed.")
        return self.interpolated_surface
    
    def detect_and_fill_gaps(self):
        """
        Détecte et comble les trous dans la surface
        """
        if self.market_data is None:
            return
        
        df = self.market_data.copy()
        
        strikes = np.sort(df['strike'].unique())
        maturities = np.sort(df['time_to_expiry'].unique())
        
        print(f"Points disponibles: {len(strikes)} strikes × {len(maturities)} maturités")
        
        strike_step = np.median(np.diff(strikes))
        maturity_step = np.median(np.diff(maturities))
        
        filled_data = []
        
        for T in maturities:
            subset = df[df['time_to_expiry'] == T]
            
            if len(subset) > 1:
                interp_1d = interpolate.interp1d(
                    subset['strike'], subset['implied_vol'],
                    kind='linear', bounds_error=False, fill_value='extrapolate'
                )
                
                for K in strikes:
                    if K not in subset['strike'].values:
                        vol = float(interp_1d(K))
                        if 0.01 < vol < 1.5:  
                            filled_data.append({
                                'strike': K,
                                'time_to_expiry': T,
                                'moneyness': K / self.spot_price,
                                'implied_vol': vol,
                                'is_filled': True
                            })
        
        if filled_data:
            filled_df = pd.DataFrame(filled_data)
            df['is_filled'] = False
            combined_df = pd.concat([df, filled_df], ignore_index=True)
            
            print(f"Points ajoutés: {len(filled_df)}")
            self.market_data = combined_df
        
        return self.market_data
    
   
            
#%%   
if __name__ == "__main__":
    test = IvBlackScholes("AAPL")
    print(test.interpolate_surface_multimethod())
    pass
# %%
