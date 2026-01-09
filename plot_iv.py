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
    ticker = "AAPL"
    iv_plotter = IvPlotIv(ticker)
    iv_plotter.plot_interactive_surface()
    iv_plotter.interpolated_surface()