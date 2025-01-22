# Monte Carlo Portfolio Simulation

A Python implementation of Monte Carlo simulation for portfolio analysis and risk assessment. This tool uses real market data to simulate potential future portfolio performance and provide comprehensive risk metrics.

## Features

- **Real Market Data Integration**
  - Fetches historical stock data from Yahoo Finance
  - Uses actual market returns and volatility
  - Accounts for asset correlations

- **Advanced Simulation Capabilities**
  - Multiple simulation paths
  - Configurable time horizons
  - Adjustable number of simulations
  - Multivariate normal distribution for returns

- **Comprehensive Analysis**
  - Portfolio value projections
  - Risk metrics calculation
  - Profit probability estimation
  - Percentile analysis
  - Value at Risk (VaR) calculations

- **Visual Analytics**
  - Simulation path visualization
  - Final value distribution plots
  - Normality assessment via Q-Q plots
  - Portfolio composition charts

## Installation

```bash
# Clone the repository
git clone https://github.com/aman-24052001/Data-Science-using-statistics/Montecarlo.git
cd monte-carlo-portfolio

# Install required packages
pip install -r requirements.txt
```

### Dependencies

```txt
numpy
pandas
yfinance
matplotlib
seaborn
scipy
tabulate
```

## Usage

### Basic Usage

```python
from portfolio_simulator import PortfolioSimulator

# Define portfolio parameters
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
weights = [0.2, 0.2, 0.2, 0.2, 0.2]
initial_investment = 10000

# Create simulator instance
simulator = PortfolioSimulator(tickers, weights, initial_investment)

# Run simulations
simulations = simulator.simulate_portfolio(num_simulations=1000)

# Analyze results
stats = simulator.analyze_simulations(simulations)

# Create visualizations
simulator.plot_simulations(simulations)
```

### Advanced Usage

#### Custom Time Horizon
```python
# Simulate for 2 years (504 trading days)
simulations = simulator.simulate_portfolio(
    num_simulations=1000, 
    time_horizon=504
)
```

#### Different Portfolio Compositions
```python
# Technology-focused portfolio
tech_tickers = ['NVDA', 'AMD', 'INTC', 'TSM', 'MU']
tech_weights = [0.3, 0.2, 0.2, 0.15, 0.15]
tech_simulator = PortfolioSimulator(tech_tickers, tech_weights, 20000)
```

## Output Examples

### Statistical Summary
```
+----------------------+------------------+
| Metric               | Value            |
+----------------------+------------------+
| Initial Investment   | $10,000.00      |
| Mean Final Value     | $12,547.83      |
| Median Final Value   | $12,234.56      |
| Std Dev Final Value  | $2,345.67       |
| 5th Percentile       | $9,123.45       |
| 95th Percentile     | $16,789.01      |
| Probability of Profit| 75.3%           |
+----------------------+------------------+
```

### Visualization Examples

The tool generates four plots:
1. **Simulation Paths**: Shows all possible portfolio value trajectories
2. **Value Distribution**: Histogram of final portfolio values
3. **Q-Q Plot**: Assessment of returns normality
4. **Portfolio Composition**: Pie chart of asset allocation

## Customization

### Simulation Parameters

```python
class PortfolioSimulator:
    def simulate_portfolio(
        self, 
        num_simulations=1000,   # Number of simulation paths
        time_horizon=252        # Trading days to simulate
    ):
        # ... simulation code ...
```

### Risk Metrics

You can modify the `analyze_simulations` method to include additional metrics:

```python
def analyze_simulations(self, simulations):
    final_values = simulations[-1, :]
    
    # Add custom metrics
    sharpe_ratio = self.calculate_sharpe_ratio(final_values)
    sortino_ratio = self.calculate_sortino_ratio(final_values)
    
    # ... additional analysis ...
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data provided by Yahoo Finance through the yfinance library
- Inspired by modern portfolio theory and risk management practices
- Built using Python's scientific computing stack

## Citation

If you use this tool in your research or project, please cite:

```bibtex
@software{monte_carlo_portfolio,
  author = {Your Name},
  title = {Monte Carlo Portfolio Simulation},
  year = {2025},
  url = {https://github.com/aman-24052001/Data-Science-using-statistics/Montecarlo}
}
```

## Disclaimer

This tool is for educational and research purposes only. It should not be used as the sole basis for making investment decisions. Always consult with a qualified financial advisor before making investment decisions.
