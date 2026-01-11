from iv_surface import IvBlackScholes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.logging_config import setup_logger
logger = setup_logger(name="iv_plot_iv", log_file=None)

class IvPlotIv(IvBlackScholes):
    def __init__(self, ticker):
        super().__init__(ticker)

        self.interpolated_surface = self.interpolate_surface_multimethod()
        self.market_data = self.detect_and_fill_gaps()

    def plot_surface_comparison(self, figsize=(16, 10)):
        """
        Visualise la comparaison entre données brutes et surface interpolée
        """
        if self.market_data is None or self.interpolated_surface is None:
            logger.error("Données de marché ou surface interpolée manquantes.")
            return
        
        fig = plt.figure(figsize=figsize)
        
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.scatter(
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
        ax1.set_xlabel('Strike')
        ax1.set_ylabel('Maturité (années)')
        ax1.set_zlabel('Volatilité implicite')
        ax1.set_title('Données Brutes du Marché')
        ax1.view_init(elev=30, azim=45)
        
        ax2 = fig.add_subplot(232, projection='3d')
        surf = ax2.plot_surface(
            self.interpolated_surface['strike_grid_2d'],
            self.interpolated_surface['maturity_grid_2d'],
            self.interpolated_surface['volatility_grid'],
            cmap='viridis',
            alpha=0.8,
            linewidth=0,
            antialiased=True
        )
        ax2.scatter(
            self.market_data['strike'],
            self.market_data['time_to_expiry'],
            self.market_data['implied_vol'],
            c='red', s=20, alpha=0.6
        )
        ax2.set_xlabel('Strike')
        ax2.set_ylabel('Maturité (années)')
        ax2.set_zlabel('Volatilité implicite')
        ax2.set_title('Surface Interpolée')
        ax2.view_init(elev=30, azim=45)
        
        ax3 = fig.add_subplot(233)
        contour = ax3.contourf(
            self.interpolated_surface['strike_grid_2d'],
            self.interpolated_surface['maturity_grid_2d'],
            self.interpolated_surface['volatility_grid'],
            levels=20,
            cmap='viridis'
        )
        ax3.scatter(
            self.market_data['strike'],
            self.market_data['time_to_expiry'],
            c='red',
            s=30,
            alpha=0.7,
            edgecolors='black'
        )
        ax3.set_xlabel('Strike')
        ax3.set_ylabel('Maturité (années)')
        ax3.set_title('Projection 2D avec Points Réels')
        plt.colorbar(contour, ax=ax3, label='Volatilité')
        
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
                    label=f'T={T:.2f} ans',
                    alpha=0.7,
                    markersize=4
                )
        ax4.axvline(self.spot_price, color='red', linestyle='--', alpha=0.5, label='Spot')
        ax4.set_xlabel('Strike')
        ax4.set_ylabel('Volatilité implicite')
        ax4.set_title('Smile par Maturité')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(235)
        moneyness_ranges = [(0.8, 0.9), (0.95, 1.05), (1.1, 1.2)]
        colors = ['blue', 'green', 'red']
        
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
                    color=colors[i],
                    label=f'K/S ∈ [{min_m}, {max_m}]',
                    alpha=0.7,
                    markersize=4
                )
        ax5.set_xlabel('Maturité (années)')
        ax5.set_ylabel('Volatilité moyenne')
        ax5.set_title('Term Structure par Moneyness')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(236)
        ax6.hist(self.market_data['implied_vol'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax6.axvline(self.market_data['implied_vol'].mean(), color='red', linestyle='--', label=f'Moyenne: {self.market_data["implied_vol"].mean():.3f}')
        ax6.set_xlabel('Volatilité implicite')
        ax6.set_ylabel('Fréquence')
        ax6.set_title('Distribution de la Volatilité')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Surface de Volatilité - {self.ticker} - Prix Spot: {self.spot_price:.2f}', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_interactive_surface(self):
        """
        Crée une visualisation interactive avec Plotly
        """
        if self.interpolated_surface is None:
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'surface'}, {'type': 'surface'}],
                [{'type': 'scatter'}, {'type': 'scatter'}],
            ],
            subplot_titles=(
                'Surface Interpolée',
                'Surface avec Points Réels',
                'Smile de Volatilité',
                'Term Structure'
            )
        )
        
        fig.add_trace(
            go.Surface(
                x=self.interpolated_surface['strike_grid_2d'],
                y=self.interpolated_surface['maturity_grid_2d'],
                z=self.interpolated_surface['volatility_grid'],
                colorscale='Viridis',
                opacity=0.9,
                name='Surface interpolée'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Surface(
                x=self.interpolated_surface['strike_grid_2d'],
                y=self.interpolated_surface['maturity_grid_2d'],
                z=self.interpolated_surface['volatility_grid'],
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
                    opacity=0.8
                ),
                name='Données réelles'
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
                        name=f'T={T:.2f} ans',
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
            title=f'Surface de Volatilité Interactive - {self.ticker}',
            scene=dict(
                xaxis_title='Strike',
                yaxis_title='Maturité (années)',
                zaxis_title='Volatilité implicite'
            ),
            scene2=dict(
                xaxis_title='Strike',
                yaxis_title='Maturité (années)',
                zaxis_title='Volatilité implicite'
            ),
            height=900,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Strike", row=2, col=1)
        fig.update_yaxes(title_text="Volatilité implicite", row=2, col=1)
        fig.update_xaxes(title_text="Maturité (années)", row=2, col=2)
        fig.update_yaxes(title_text="Volatilité implicite", row=2, col=2)
        
        fig.show()
        
        return fig

if __name__ == "__main__":
    ticker = "MSFT"
    iv_plotter = IvPlotIv(ticker)
    iv_plotter.plot_surface_comparison()
    iv_plotter.plot_interactive_surface()