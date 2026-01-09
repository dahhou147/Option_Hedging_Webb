# LaTeX Paper Documentation

This directory contains a comprehensive LaTeX paper documenting the options hedging strategy implementation.

## Files

- `paper.tex`: Main LaTeX document
- `generate_paper_figures.py`: Python script to generate all figures for the paper
- `compile_paper.sh`: Bash script to compile the LaTeX paper
- `figures/`: Directory containing generated figures (created automatically)

## Generating the Paper

### Step 1: Generate Figures

First, generate all the figures referenced in the paper:

```bash
python generate_paper_figures.py
```

This will create the following figures in the `figures/` directory:
- `volatility_smile.png`: Implied volatility smile
- `cumulative_pnl.png`: Cumulative PnL over time
- `hedge_ratios.png`: Evolution of hedge ratios
- `pnl_distribution.png`: Distribution of final PnL
- `price_paths.png`: Monte Carlo simulated price paths
- `drawdown.png`: Drawdown analysis

### Step 2: Compile LaTeX Paper

#### Option A: Using the provided script

```bash
./compile_paper.sh
```

#### Option B: Manual compilation

```bash
pdflatex paper.tex
pdflatex paper.tex
```

The compiled PDF will be saved as `paper.pdf`.

## Requirements

### LaTeX Packages

The paper requires the following LaTeX packages (usually included in standard distributions):
- `amsmath`, `amsfonts`, `amssymb`: Mathematical symbols
- `graphicx`: For including figures
- `hyperref`: For hyperlinks
- `booktabs`: For professional tables
- `geometry`: For page layout
- `float`: For figure placement
- `subcaption`: For subfigures
- `listings`: For code listings
- `xcolor`: For colors
- `natbib`: For bibliography

### Python Dependencies

All Python dependencies should already be in `requirements.txt`. The figure generation script uses:
- numpy
- matplotlib
- seaborn
- pandas
- yfinance
- scipy

## Paper Structure

1. **Introduction**: Overview of the problem and contributions
2. **Theoretical Background**: 
   - Black-Scholes model
   - The Greeks
   - Girsanov's theorem
3. **Methodology**: 
   - Multi-Greek hedging strategy
   - Market data calibration
   - Dynamic hedging implementation
4. **Implementation**: Code structure and algorithms
5. **Results and Analysis**: 
   - Performance metrics
   - Visualizations
   - Discussion
6. **Conclusion**: Summary and future work

## Customization

### Changing the Ticker

To analyze a different stock, edit `generate_paper_figures.py` and change:

```python
TICKER = "YOUR_TICKER"  # e.g., "AAPL", "GOOGL", "TSLA"
```

### Adjusting Simulation Parameters

Modify the parameters in `generate_paper_figures.py`:

```python
N = 252  # Number of time steps (daily rebalancing)
M = 100  # Number of Monte Carlo paths
```

### Modifying Paper Content

Edit `paper.tex` to customize:
- Title and author information
- Abstract
- Section content
- Bibliography entries

## Troubleshooting

### LaTeX Compilation Errors

If you encounter LaTeX errors:
1. Ensure all required packages are installed
2. Check that all figures exist in the `figures/` directory
3. Verify that figure paths in `paper.tex` are correct

### Figure Generation Errors

If figure generation fails:
1. Check that market data is available for the ticker
2. Verify internet connection (for yfinance data)
3. The script includes fallback synthetic data generation

### Missing Dependencies

Install missing Python packages:
```bash
pip install -r requirements.txt
```

## Notes

- The paper uses a standard academic format suitable for submission
- All figures are generated at 300 DPI for high-quality printing
- The bibliography uses a simple format; you can switch to BibTeX for more advanced citation management
- The paper assumes familiarity with options pricing theory

## License

Same as the main project (MIT License).

