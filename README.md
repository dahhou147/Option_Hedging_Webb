# Options Hedging Strategy

This project implements an advanced options hedging strategy using the Black-Scholes model. The strategy aims to neutralize delta, gamma, and vega risks in an options portfolio.

## Features

- **Option Pricing**: Implementation of the Black-Scholes model for option pricing
- **Greeks Calculation**: Delta, Gamma, Vega, Theta
- **Multi-Greek Hedging**: Simultaneous neutralization of delta, gamma, and vega risks
- **Monte Carlo Simulation**: Price path generation under risk-neutral measure
- **Performance Analysis**: Performance metrics and visualizations
- **Model Calibration**: Market data retrieval and parameter calibration

## Code Structure

### Main Modules

1. **pricing_model.py**
   - `BlackScholesPricer`: European option pricing
   - `Greeks`: Greeks calculation
   - `ConstructPortfolio`: Hedging portfolio construction
   - `GirsanovSimulator`: Price path simulation
   - `VolatilitySmile`: Implied volatility surface calculation

2. **calibration.py**
   - `GetMarketData`: Market data retrieval and processing
   - Model parameter calibration
   - Options data management

3. **launch_simulation.py**
   - `Launcher`: Main interface for running simulations
   - Portfolio performance analysis
   - Results visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dahhou147/Option_Hedging_Webb.git
cd Option_Hedging_Webb
```

2. Create and activate the virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from simulations.launch_simulation import Launcher

# Initialize the launcher with a ticker
launcher = Launcher("MSFT")

# Launch the simulation
portfolio = launcher.launch()

# Analyze results
final_pnl = portfolio.pnl[-1, :]
```

## Dependencies

- NumPy >= 1.24.3
- SciPy >= 1.10.1
- Matplotlib >= 3.7.1
- Pandas >= 2.0.2
- yfinance >= 0.2.18
- seaborn

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

## Author

Oussama Dahhou

## License

This project is licensed under the MIT License.
