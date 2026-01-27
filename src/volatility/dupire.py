"""
Dupire's formula is a formula for the implied volatility of an option.
It is used to reconstruct the dynamic of the underlying asset's volatility.
"""

from .iv_surface import IvBlackScholes   

class Dupire(IvBlackScholes):
    def __init__(self, ticker):
        super().__init__(ticker)
        self.market_data = self.detect_and_fill_gaps()
        self.surface = self.interpolate_surface_multimethod()
    
    def compute_greeks(self):
        """
        Compute the Greeks of the option.
        """
        pass

if __name__ == "__main__":
    dupire = Dupire("AAPL")
    print(dupire.market_data)