import scipy.optimize as so
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay
import pandas as pd
import numpy as np
from pricer.pricing_model import BlackScholesPricer
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
        self.dividend_yield = self.get_dividend_yield()
    
    def get_free_rate(self):
        df = obb.economy.interest_rates(country="USA", maturity="10Y").to_df()
        if not df.empty:
            return df["value"].iloc[-1] 
        else:
            logger.warning("10-year US Treasury rate not found, defaulting to 0.03")
            return 0.03
    
    def get_dividend_yield(self):
        """Récupère le dividend yield si disponible"""
        try:
            info = obb.equity.profile(symbol=self.ticker).to_df()
            if 'dividendYield' in info.columns:
                return float(info['dividendYield'].iloc[0]) / 100.0 if not pd.isna(info['dividendYield'].iloc[0]) else 0.0
        except:
            logger.warning("Dividend yield not found, defaulting to 0.0")
            return 0.0


class IvBlackScholes(DataManager):
    def __init__(self, ticker, recalculate_iv=True, min_volume=10, max_spread_ratio=0.5):
        super().__init__(ticker=ticker)
        self.recalculate_iv = recalculate_iv
        self.min_volume = min_volume
        self.max_spread_ratio = max_spread_ratio
        self.processed_options = self.preprocessing()
        self.market_data = self.apply_kernel()
        self.interpolated_surface = None
        self.coordinates = {}

    def preprocessing(self):
        """Préprocessing amélioré avec filtrage par volume et spread"""
        df = self.options.copy()
        df["time_to_expiry"] = (
            pd.to_datetime(df["expiration"])
            - pd.to_datetime(df.get("as_of_date", pd.Timestamp.today()))
        ).dt.days / 365
        
        df_filtered = df[df["time_to_expiry"] > 0.25].reset_index(drop=True)
        
        if 'volume' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['volume'] >= self.min_volume]
            logger.info(f"Filtered by volume >= {self.min_volume}: {len(df_filtered)} options remaining")
        
        if 'bid' in df_filtered.columns and 'ask' in df_filtered.columns:
            df_filtered = df_filtered[
                (df_filtered['ask'] - df_filtered['bid']) / df_filtered['ask'] <= self.max_spread_ratio
            ]
            logger.info(f"Filtered by spread <= {self.max_spread_ratio*100}%: {len(df_filtered)} options remaining")
        
        price_col = (
            "last_trade_price"
            if "last_trade_price" in df_filtered.columns
            else ("lastPrice" if "lastPrice" in df_filtered.columns else None)
        )
        if price_col is None:
            raise KeyError(
                "Market price not found (no 'last_trade_price' or 'lastPrice')"
            )
        
        df_filtered = df_filtered[df_filtered[price_col] > 0.01]
        
        columns_to_select = ["strike", "time_to_expiry", "option_type", price_col]
        if "implied_volatility" in df_filtered.columns and not self.recalculate_iv:
            columns_to_select.append("implied_volatility")
        
        df_option = df_filtered.groupby(
            ["strike", "time_to_expiry"], as_index=False
        ).first()[columns_to_select]
        df_option = df_option.rename(columns={price_col: "market_price"})
        
        logger.info(f"Preprocessed {len(df_option)} option contracts")
        return df_option

    def get_implied_vol(self, strike, time_to_expiry, market_price, option_type):
        """Calcul amélioré avec gestion des dividendes"""
        def objective_function(volatility):
            pricer = BlackScholesPricer(
                S0=self.spot_price,
                K=strike,
                T=time_to_expiry,
                sigma=volatility,
                r=self.free_rate,
                q=self.dividend_yield, 
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

        except Exception as e:
            logger.debug(f"Error calculating IV for K={strike}, T={time_to_expiry}: {e}")
            return np.nan

    def apply_kernel(self):
        """Application du kernel avec option de recalcul"""
        df = self.processed_options.copy()
        
        if "implied_volatility" in df.columns and not self.recalculate_iv:
            df["implied_vol"] = df["implied_volatility"]
            df = df[(df["implied_vol"] > 0.01) & (df["implied_vol"] < 1.5)]
            logger.info("Using existing implied volatility values")
        else:
            df["implied_vol"] = df.apply(
                lambda row: self.get_implied_vol(
                    row["strike"],
                    row["time_to_expiry"],
                    row["market_price"],
                    row["option_type"],
                ),
                axis=1,
            )
            logger.info("Recalculated implied volatility values")
        
        df = df[~df["implied_vol"].isna()]
        
        return df.assign(spot_price=self.spot_price).assign(
            moneyness=lambda x: x["strike"] / x["spot_price"]
        ).assign(
            log_moneyness=lambda x: np.log(x["strike"] / x["spot_price"])
        )

    def validate_data(self):
        """Valide la qualité des données avant interpolation"""
        df = self.market_data
        if df is None or len(df) == 0:
            logger.error("No market data available")
            return False
        
        nan_count = df['implied_vol'].isna().sum()
        if nan_count > len(df) * 0.1:
            logger.error(f"Too many NaN values: {nan_count}/{len(df)}")
            return False
        
        extreme_low = (df['implied_vol'] < 0.001).sum()
        extreme_high = (df['implied_vol'] > 2.0).sum()
        if extreme_low > 0 or extreme_high > 0:
            logger.warning(f"Extreme volatility values: {extreme_low} too low, {extreme_high} too high")
        
        strike_coverage = (df['strike'].max() - df['strike'].min()) / self.spot_price
        if strike_coverage < 0.2:
            logger.warning(f"Limited strike coverage: {strike_coverage:.2%}")
        
        if len(df) < 10:
            logger.error(f"Insufficient data points: {len(df)}")
            return False
        
        logger.info("Data validation passed")
        return True

    def check_arbitrage(self, surface_dict):
        """Vérifie les contraintes d'arbitrage sur la surface interpolée"""
        strikes = surface_dict['strikes_grid']
        maturities = surface_dict['maturities_grid']
        vol_grid = surface_dict['volatility_grid']
        
        for i, T in enumerate(maturities):
            vol_slice = vol_grid[i, :]
            if len(vol_slice) < 3:
                continue
            second_diff = np.diff(vol_slice, n=2)
            if np.any(second_diff < -1e-4):  
                logger.warning(f"Possible butterfly arbitrage detected at T={T:.2f}")
                return False
        
        atm_idx = np.argmin(np.abs(strikes - self.spot_price))
        if atm_idx < len(strikes):
            atm_term = vol_grid[:, atm_idx]
            if len(atm_term) > 1:
                term_diff = np.diff(atm_term)
                if np.any(term_diff < -0.15):  
                    logger.warning("Possible calendar arbitrage detected")
                    return False
        
        logger.info("Arbitrage check passed")
        return True

    def interpolate_surface_multimethod(self, grid_size=50):
        """
        Interpolates the volatility surface using multiple methods
        with improved error handling and validation
        """
        if not self.validate_data():
            logger.error("Data validation failed, cannot interpolate")
            return None

        df = self.market_data.copy()

        x_min, x_max = df["strike"].min(), df["strike"].max()
        y_min, y_max = df["time_to_expiry"].min(), df["time_to_expiry"].max()

        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1

        x_grid = np.linspace(max(0, x_min - x_padding), x_max + x_padding, grid_size)
        y_grid = np.linspace(max(0.01, y_min - y_padding), y_max + y_padding, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid, indexing='xy')

        points = df[["strike", "time_to_expiry"]].values
        values = df["implied_vol"].values
        
        data_density = len(df) / ((x_max - x_min) * (y_max - y_min))
        if data_density > 10:
            sigma = 0.5
        elif data_density > 5:
            sigma = 1.0
        else:
            sigma = 1.5
        logger.info(f"Data density: {data_density:.2f}, using sigma={sigma} for smoothing")

        logger.info("RBF Interpolation...")
        try:
            rbf = interpolate.RBFInterpolator(
                points, values, kernel="thin_plate_spline"
            )
            grid_points = np.column_stack((X.ravel(), Y.ravel()))
            Z_rbf = rbf(grid_points).reshape(X.shape)
        except Exception as e:
            logger.warning(f"RBF interpolation failed: {e}")
            Z_rbf = np.full(X.shape, np.nan)

        logger.info("Linear interpolation...")
        try:
            tri = Delaunay(points)
            interp_linear = interpolate.LinearNDInterpolator(tri, values)
            Z_linear = interp_linear(X, Y)
        except Exception as e:
            logger.warning(f"Linear interpolation failed: {e}")
            Z_linear = np.full(X.shape, np.nan)

        logger.info("Spline interpolation...")
        try:
            x_unique = np.unique(points[:, 0])
            y_unique = np.unique(points[:, 1])

            if len(x_unique) > 3 and len(y_unique) > 3:
                Z_spline = interpolate.griddata(
                    points, values, (X, Y), method="cubic", fill_value=np.nan
                )
            else:
                Z_spline = np.full(X.shape, np.nan)
        except Exception as e:
            logger.warning(f"Spline interpolation failed: {e}")
            Z_spline = np.full(X.shape, np.nan)

        Z_combined = np.copy(Z_linear)
        nan_mask = np.isnan(Z_combined)
        if not np.all(np.isnan(Z_rbf)):
            Z_combined[nan_mask] = Z_rbf[nan_mask]

        nan_mask = np.isnan(Z_combined)
        if not np.all(np.isnan(Z_spline)):
            Z_combined[nan_mask] = Z_spline[nan_mask]

        Z_smoothed = gaussian_filter(Z_combined, sigma=sigma)

        Z_smoothed = np.clip(Z_smoothed, 0.01, 1.5)

        try:
            if not np.allclose(X[0, :], x_grid):
                logger.debug("Grid orientation mismatch detected, transposing")
                X = X.T
                Y = Y.T
                Z_smoothed = Z_smoothed.T
        except Exception as e:
            logger.exception("Error while validating grid orientation")

        self.interpolated_surface = {
            "strikes_grid": x_grid,
            "maturities_grid": y_grid,
            "strike_grid_2d": X,
            "maturity_grid_2d": Y,
            "volatility_grid": Z_smoothed,
            "original_points": points,
            "original_values": values,
        }

        if not self.check_arbitrage(self.interpolated_surface):
            logger.warning("Surface may violate arbitrage constraints")

        logger.info("Interpolation completed.")
        return self.interpolated_surface
    
    def detect_and_fill_gaps(self, max_extrapolation_ratio=0.1):
        """
        Détecte et remplit les gaps avec limitation de l'extrapolation
        """
        if self.market_data is None:
            return None
        
        df = self.market_data.copy()
        
        strikes = np.sort(df['strike'].unique())
        maturities = np.sort(df['time_to_expiry'].unique())
        
        logger.info("Available data points: %d strikes × %d maturities", len(strikes), len(maturities))
        
        filled_data = []
        
        for T in maturities:
            subset = df[df['time_to_expiry'] == T]
            
            if len(subset) > 1:
                strike_range = subset['strike'].max() - subset['strike'].min()
                min_strike = subset['strike'].min()
                max_strike = subset['strike'].max()
                
                interp_1d = interpolate.interp1d(
                    subset['strike'], subset['implied_vol'],
                    kind='linear', bounds_error=False, fill_value=np.nan
                )
                
                for K in strikes:
                    if K not in subset['strike'].values:
                        if K < min_strike - max_extrapolation_ratio * strike_range:
                            continue
                        if K > max_strike + max_extrapolation_ratio * strike_range:
                            continue
                        
                        vol = float(interp_1d(K))
                        if not np.isnan(vol) and 0.01 < vol < 1.5:
                            filled_data.append({
                                'strike': K,
                                'time_to_expiry': T,
                                'moneyness': K / self.spot_price,
                                'log_moneyness': np.log(K / self.spot_price),
                                'implied_vol': vol,
                                'is_filled': True
                            })
        
        if filled_data:
            filled_df = pd.DataFrame(filled_data)
            df['is_filled'] = False
            combined_df = pd.concat([df, filled_df], ignore_index=True)
            logger.info(f"Data points added: {len(filled_df)}")
            self.market_data = combined_df
        
        return self.market_data

