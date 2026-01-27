from .iv_surface import IvBlackScholes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config.logging_config import setup_logger
logger = setup_logger(name="iv_plot_iv", log_file=None)


class IvPlotIv(IvBlackScholes):
    def __init__(self, ticker):
        super().__init__(ticker)
        self.market_data = self.detect_and_fill_gaps()
        self.interpolated_surface = self.interpolate_surface_multimethod()
        
        if self.interpolated_surface is not None:
            self.validate_surface_dimensions()

    def validate_surface_dimensions(self):
        """Vérifie que les dimensions de la surface sont cohérentes"""
        surface = self.interpolated_surface
        if surface is None:
            return False
        
        X = surface['strike_grid_2d']
        Y = surface['maturity_grid_2d']
        Z = surface['volatility_grid']
        
        if X.shape != Y.shape or X.shape != Z.shape:
            logger.error(f"Dimension mismatch: X={X.shape}, Y={Y.shape}, Z={Z.shape}")
            return False
        
        logger.info(f"Surface dimensions validated: {X.shape}")
        return True

    def validate_plot_data(self):
        """Valide que les données de plot sont cohérentes"""
        if self.interpolated_surface is None:
            return False
        
        strikes = self.market_data['strike']
        maturities = self.market_data['time_to_expiry']
        
        strike_grid = self.interpolated_surface['strikes_grid']
        maturity_grid = self.interpolated_surface['maturities_grid']
        
        strike_in_range = (strikes.min() >= strike_grid.min()) and (strikes.max() <= strike_grid.max())
        maturity_in_range = (maturities.min() >= maturity_grid.min()) and (maturities.max() <= maturity_grid.max())
        
        if not strike_in_range:
            logger.warning(f"Some strikes are outside the grid range: [{strikes.min():.2f}, {strikes.max():.2f}] vs [{strike_grid.min():.2f}, {strike_grid.max():.2f}]")
        if not maturity_in_range:
            logger.warning(f"Some maturities are outside the grid range: [{maturities.min():.3f}, {maturities.max():.3f}] vs [{maturity_grid.min():.3f}, {maturity_grid.max():.3f}]")
        
        return True

    def test_surface_orientation(self):
        """
        Teste visuellement l'orientation de la surface
        Crée un plot simple pour vérifier que tout est correct
        """
        if self.interpolated_surface is None:
            logger.error("Interpolated surface is None")
            return False
        
        surface = self.interpolated_surface
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        X = surface['strike_grid_2d']
        Y = surface['maturity_grid_2d']
        Z = surface['volatility_grid']
        
        first_maturity_idx = 0
        strikes_slice = X[first_maturity_idx, :]
        vol_slice = Z[first_maturity_idx, :]
        
        axes[0].plot(strikes_slice, vol_slice, 'o-', markersize=4)
        axes[0].set_xlabel('Strike ($)')
        axes[0].set_ylabel('Implied Volatility')
        axes[0].set_title(f'Volatility Smile at T={Y[first_maturity_idx, 0]:.2f} years')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(self.spot_price, color='red', linestyle='--', alpha=0.5, label='Spot')
        axes[0].legend()
 
        atm_idx = np.argmin(np.abs(X[0, :] - self.spot_price))
        maturities_slice = Y[:, atm_idx]
        vol_term = Z[:, atm_idx]
        
        axes[1].plot(maturities_slice, vol_term, 's-', color='green', markersize=4)
        axes[1].set_xlabel('Maturity (years)')
        axes[1].set_ylabel('Implied Volatility')
        axes[1].set_title(f'Term Structure at K={X[0, atm_idx]:.2f}')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Surface Orientation Test - {self.ticker}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        strikes_increasing = np.all(np.diff(strikes_slice) > 0)
        maturities_increasing = np.all(np.diff(maturities_slice) > 0)
        
        if not strikes_increasing:
            logger.warning("Strikes are not monotonically increasing!")
        if not maturities_increasing:
            logger.warning("Maturities are not monotonically increasing!")
        
        return strikes_increasing and maturities_increasing

    def plot_surface_comparison(self, figsize=(16, 10)):
        """
        Visualize the comparison between raw data and interpolated surface
        Version améliorée avec validation
        """
        if self.market_data is None or self.interpolated_surface is None:
            logger.error("Market data or interpolated surface missing.")
            return None
        
        if not self.validate_plot_data():
            logger.warning("Plot data validation failed, but continuing...")
        
        fig = plt.figure(figsize=figsize)
        
        ax1 = fig.add_subplot(231, projection='3d')
        scatter1 = ax1.scatter(
            self.market_data['strike'],
            self.market_data['time_to_expiry'],
            self.market_data['implied_vol'],
            c=self.market_data['implied_vol'],
            cmap='viridis',
            s=50,
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5
        )
        ax1.set_xlabel('Strike ($)')
        ax1.set_ylabel('Maturity (years)')
        ax1.set_zlabel('Implied Volatility')
        ax1.set_title('Raw Market Data')
        ax1.view_init(elev=30, azim=45)
        plt.colorbar(scatter1, ax=ax1, label='Volatility', shrink=0.8)
        
        ax2 = fig.add_subplot(232, projection='3d')
        X = self.interpolated_surface['strike_grid_2d']
        Y = self.interpolated_surface['maturity_grid_2d']
        Z = self.interpolated_surface['volatility_grid']
        
        surf = ax2.plot_surface(
            X, Y, Z,
            cmap='viridis',
            alpha=0.8,
            linewidth=0,
            antialiased=True
        )
        ax2.scatter(
            self.market_data['strike'],
            self.market_data['time_to_expiry'],
            self.market_data['implied_vol'],
            c='red', s=20, alpha=0.6, label='Market data'
        )
        ax2.set_xlabel('Strike ($)')
        ax2.set_ylabel('Maturity (years)')
        ax2.set_zlabel('Implied Volatility')
        ax2.set_title('Interpolated Surface')
        ax2.view_init(elev=30, azim=45)
        plt.colorbar(surf, ax=ax2, label='Volatility', shrink=0.8)
        
        ax3 = fig.add_subplot(233)
        contour = ax3.contourf(
            X, Y, Z,
            levels=20,
            cmap='viridis'
        )
        ax3.scatter(
            self.market_data['strike'],
            self.market_data['time_to_expiry'],
            c='red',
            s=30,
            alpha=0.7,
            edgecolors='black',
            label='Market data'
        )
        ax3.set_xlabel('Strike ($)')
        ax3.set_ylabel('Maturity (years)')
        ax3.set_title('2D Projection with Real Data Points')
        ax3.legend()
        plt.colorbar(contour, ax=ax3, label='Volatility')
        
        ax4 = fig.add_subplot(234)
        unique_maturities = np.sort(self.market_data['time_to_expiry'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, min(5, len(unique_maturities))))
        
        for i, T in enumerate(unique_maturities[:5]):
            subset = self.market_data[self.market_data['time_to_expiry'] == T]
            if len(subset) > 2:
                subset = subset.sort_values('strike')
                ax4.plot(
                    subset['strike'],
                    subset['implied_vol'],
                    'o-',
                    color=colors[i],
                    label=f'T={T:.2f} years',
                    alpha=0.7,
                    markersize=4
                )
        ax4.axvline(self.spot_price, color='red', linestyle='--', alpha=0.5, label='Spot')
        ax4.set_xlabel('Strike ($)')
        ax4.set_ylabel('Implied Volatility')
        ax4.set_title('Volatility Smile by Maturity')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(235)
        moneyness_ranges = [(0.8, 0.9), (0.95, 1.05), (1.1, 1.2)]
        colors_moneyness = ['blue', 'green', 'red']
        
        for i, (min_m, max_m) in enumerate(moneyness_ranges):
            subset = self.market_data[
                (self.market_data['moneyness'] >= min_m) & 
                (self.market_data['moneyness'] <= max_m)
            ]
            if len(subset) > 2:
                term_data = subset.groupby('time_to_expiry')['implied_vol'].mean().reset_index()
                term_data = term_data.sort_values('time_to_expiry')
                ax5.plot(
                    term_data['time_to_expiry'],
                    term_data['implied_vol'],
                    's-',
                    color=colors_moneyness[i],
                    label=f'K/S ∈ [{min_m}, {max_m}]',
                    alpha=0.7,
                    markersize=4
                )
        ax5.set_xlabel('Maturity (years)')
        ax5.set_ylabel('Average Volatility')
        ax5.set_title('Term Structure by Moneyness')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(236)
        ax6.hist(self.market_data['implied_vol'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        mean_vol = self.market_data['implied_vol'].mean()
        ax6.axvline(mean_vol, color='red', linestyle='--', label=f'Mean: {mean_vol:.3f}')
        ax6.set_xlabel('Implied Volatility')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Volatility Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Volatility Surface - {self.ticker} - Spot Price: ${self.spot_price:.2f}', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_interactive_surface(self):
        """
        Create an interactive visualization with Plotly
        Version améliorée avec validation et meilleure configuration
        """
        if self.interpolated_surface is None:
            logger.error("Interpolated surface is None")
            return None
        
        if not self.validate_surface_dimensions():
            logger.error("Surface dimensions are inconsistent")
            return None
        
        surface = self.interpolated_surface
        
        X = surface['strike_grid_2d']
        Y = surface['maturity_grid_2d']
        Z = surface['volatility_grid']
        
        logger.debug(f"Grid shapes: X={X.shape}, Y={Y.shape}, Z={Z.shape}")
        logger.debug(f"X range: [${X.min():.2f}, ${X.max():.2f}]")
        logger.debug(f"Y range: [{Y.min():.3f}, {Y.max():.3f}] years")
        logger.debug(f"Z range: [{Z.min():.3f}, {Z.max():.3f}]")
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'surface'}, {'type': 'surface'}],
                [{'type': 'scatter'}, {'type': 'scatter'}],
            ],
            subplot_titles=(
                'Interpolated Surface',
                'Surface with Real Data Points',
                'Volatility Smile',
                'Term Structure'
            )
        )
        
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale='Viridis',
                opacity=0.9,
                name='Interpolated surface',
                colorbar=dict(title="Volatility", len=0.4, y=0.75, x=1.02)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale='Viridis',
                opacity=0.7,
                name='Surface',
                showscale=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter3d(
                x=self.market_data['strike'],
                y=self.market_data['time_to_expiry'],
                z=self.market_data['implied_vol'],
                mode='markers',
                marker=dict(
                    size=5,
                    color='red',
                    opacity=0.8,
                    line=dict(width=1, color='darkred')
                ),
                name='Real data'
            ),
            row=1, col=2
        )
        
        unique_maturities = np.sort(self.market_data['time_to_expiry'].unique())
        
        for i, T in enumerate(unique_maturities[:3]):
            subset = self.market_data[self.market_data['time_to_expiry'] == T]
            if len(subset) > 2:
                subset = subset.sort_values('strike')
                fig.add_trace(
                    go.Scatter(
                        x=subset['strike'],
                        y=subset['implied_vol'],
                        mode='lines+markers',
                        name=f'T={T:.2f} years',
                        line=dict(width=2),
                        marker=dict(size=6)
                    ),
                    row=2, col=1
                )
        
        atm_subset = self.market_data[
            (self.market_data['moneyness'] >= 0.98) & 
            (self.market_data['moneyness'] <= 1.02)
        ]
        
        if len(atm_subset) > 2:
            term_data = atm_subset.groupby('time_to_expiry')['implied_vol'].mean().reset_index()
            term_data = term_data.sort_values('time_to_expiry')
            
            fig.add_trace(
                go.Scatter(
                    x=term_data['time_to_expiry'],
                    y=term_data['implied_vol'],
                    mode='lines+markers',
                    name='ATM Term Structure',
                    line=dict(width=2, color='green'),
                    marker=dict(size=8)
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f'Interactive Volatility Surface - {self.ticker}',
            scene=dict(
                xaxis_title='Strike ($)',
                yaxis_title='Maturity (years)',
                zaxis_title='Implied Volatility',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            scene2=dict(
                xaxis_title='Strike ($)',
                yaxis_title='Maturity (years)',
                zaxis_title='Implied Volatility',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            height=900,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Strike ($)", row=2, col=1)
        fig.update_yaxes(title_text="Implied Volatility", row=2, col=1)
        fig.update_xaxes(title_text="Maturity (years)", row=2, col=2)
        fig.update_yaxes(title_text="Implied Volatility", row=2, col=2)
        
        fig.show()
        
        return fig
    
    def run(self):
        self.test_surface_orientation()
        self.plot_surface_comparison()
        self.plot_interactive_surface()

if __name__ == "__main__":
    ticker = "MSFT"
    iv_plotter = IvPlotIv(ticker)
    iv_plotter.run()

