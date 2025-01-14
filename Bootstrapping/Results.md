# Bootstrap Resampling Implementation from Scratch

## Overview
This repository contains a scratch implementation of Bootstrap Resampling, a powerful statistical technique used in data science for:
- Estimating the uncertainty of statistical measures
- Creating confidence intervals
- Performing hypothesis testing without assuming normal distribution
- Making reliable decisions with limited data

## Why Bootstrap?
Traditional statistical methods often assume normal distribution and require large sample sizes. Bootstrap overcomes these limitations by:
- Working with any distribution
- Providing robust estimates with smaller samples
- Enabling estimation of complex statistics
- Supporting decision making with confidence intervals

## Project Structure
```
ML_scratch-implementation/
├── bootstrap_analysis.py     # Main implementation file
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies
```


## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from bootstrap_analysis import BootstrapAnalyzer

# Initialize analyzer
analyzer = BootstrapAnalyzer(data, n_iterations=1000)

# Get confidence interval for mean
ci_mean = analyzer.get_confidence_interval('column_name', statistic='mean')

# Perform hypothesis test
p_value = analyzer.hypothesis_test(group1, group2, statistic='mean')
```

## Features
1. Bootstrap Resampling Implementation
   - Random sampling with replacement
   - Support for various statistics (mean, median, etc.)
   - Customizable number of iterations

2. Statistical Analysis
   - Confidence interval estimation
   - Hypothesis testing
   - Effect size calculation

3. Visualization
   - Bootstrap distribution plots
   - Confidence interval visualization
   - Comparison plots for hypothesis tests
