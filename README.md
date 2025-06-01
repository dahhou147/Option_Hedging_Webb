# Options Hedging Strategy

This project implements an advanced options hedging strategy using the Black-Scholes model. The strategy aims to neutralize delta, gamma, and vega risks in an options portfolio.

## Features

- **Option Pricing**: Implementation of the Black-Scholes model for option pricing
- **Greeks Calculation**: Delta, Gamma, Vega, Theta
- **Multi-Greek Hedging**: Simultaneous neutralization of delta, gamma, and vega
- **Monte Carlo Simulation**: Price path generation under risk-neutral measure
- **Performance Analysis**: Performance metrics and visualizations

## Code Structure

### Main Classes

1. **BlackScholesPricer**
   - European option pricing
   - d1 and d2 parameters calculation
   - Support for call and put options

2. **Greeks**
   - Greeks calculation (delta, gamma, vega, theta)
   - Support for call and put options

3. **ConstructPortfolio**
   - Hedging portfolio construction
   - Delta, gamma, and vega risk neutralization
   - Dynamic portfolio management

4. **GirsanovSimulator**
   - Price path simulation under risk-neutral measure
   - Implementation of Girsanov's theorem

### Usage Example

```python
# Market parameters
S0 = 100.0  # Initial price
K = 100.0   # Strike price
T = 1.0     # Time to maturity
r = 0.05    # Risk-free rate
sigma = 0.2 # Volatility
N = 252     # Number of time steps
M = 100     # Number of simulations

# Create price paths
simulator = GirsanovSimulator(S0, mu, r, sigma, N, T, M)
paths = simulator.generate_paths()

# Create hedging portfolio
pricer = BlackScholesPricer(S0, K, T, sigma, r)
portfolio = ConstructPortfolio(pricer, paths, N, T, K*0.9, K*1.1)

# Execute hedging
portfolio.hedge_portfolio(option_type="call")
```

## Installation

```bash
pip install -r requirements.txt
```

## Dependencies

- NumPy
- SciPy
- Matplotlib
- yfinance
- seaborn
- dataclasses

## Advanced Features

1. **Multi-Greek Hedging**
   - Use of two hedging options with different strikes
   - Underlying position for complete neutralization

2. **Risk Management**
   - Coefficient regularization to avoid extreme positions
   - Numerical error handling

3. **Performance Analysis**
   - PnL distribution
   - Portfolio value evolution
   - Cash position tracking
   - Performance metrics (Sharpe ratio, etc.)

## Future Improvements

- [ ] Add transaction costs
- [ ] Support for American options
- [ ] Implementation of alternative volatility models
- [ ] Hedging parameter optimization
- [ ] Historical data backtesting

## References

- John Hull's book on options, futures and other derivatives
- Surface de volatilité, Peter TANKOV Paris Diderot University
- Copulas: Sas Documentation, : https://support.sas.com/documentation/onlinedoc/ets/132/copula.pdf

## Author

Oussama Dahhou

## License

This project is licensed under the MIT License.
