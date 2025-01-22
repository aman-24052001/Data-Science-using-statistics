# Markov Chain Analysis Tool

A comprehensive Python function for analyzing and visualizing Markov Chains with real-world applications in healthcare, supply chain management, and web ranking.

## Features

- **Multiple Application Domains**
  - Healthcare: Patient condition progression modeling
  - Supply Chain: Inventory state transitions
  - Web Ranking: Simplified PageRank implementation

- **Analysis Capabilities**
  - Steady state calculation
  - N-step predictions
  - Absorption probabilities
  - Convergence analysis

- **Visualization Tools**
  - Network diagrams
  - Transition probability matrices
  - State evolution plots
  - Convergence visualization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/markov-chain-analysis.git
cd markov-chain-analysis

# Install required packages
pip install -r requirements.txt
```

### Dependencies

```txt
numpy
pandas
matplotlib
seaborn
networkx
scipy
tabulate
scikit-learn
```

## Usage

### Basic Usage

```python
from markov_chain_analyzer import MarkovChainAnalyzer

# Create analyzer instance
analyzer = MarkovChainAnalyzer()

# Analyze specific application
report, visualization = analyzer.generate_report('healthcare')
```

### Custom Analysis

```python
# Define custom transition matrix
custom_states = ['State1', 'State2', 'State3']
custom_matrix = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])

# Analyze custom Markov chain
steady_state = analyzer.calculate_steady_state(custom_matrix)
predictions = analyzer.predict_n_steps(custom_matrix, initial_state, n_steps=10)
```

## Applications

### 1. Healthcare Analysis
- Models patient condition progression
- Uses breast cancer dataset for transition probabilities
- Predicts condition state probabilities

### 2. Supply Chain Management
- Models inventory level transitions
- Analyzes stock level probabilities
- Predicts future stock states

### 3. Web Page Ranking
- Implements simplified PageRank algorithm
- Models web page importance
- Analyzes page visit probabilities

## Output Examples

### 1. Analysis Report
```
Markov Chain Analysis Report - Healthcare
======================================
1. Initial State Distribution:
+------------+-------------+
| State      | Probability |
+------------+-------------+
| Normal     | 0.333      |
| Benign     | 0.333      |
| Malignant  | 0.333      |
+------------+-------------+

2. Steady State Distribution:
...
```

### 2. Visualizations
- Network diagrams showing state transitions
- Probability evolution plots
- Convergence analysis graphs

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewApplication`)
3. Commit your changes (`git commit -m 'Add new application'`)
4. Push to the branch (`git push origin feature/NewApplication`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{markov_chain_analysis,
  author = {Your Name},
  title = {Markov Chain Analysis Tool},
  year = {2025},
  url = {https://github.com/aman-24052001/Data-Science-using-statistics/markov-chain}
}
```

## Acknowledgments

- Scikit-learn for providing datasets
- NetworkX for network visualizations
- Academic literature on Markov Chain applications
