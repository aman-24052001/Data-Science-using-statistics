import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

class PortfolioSimulator:
    def __init__(self, tickers, weights=None, initial_investment=10000):
        """
        Initialize portfolio simulator with stock tickers and weights
        
        Parameters:
        tickers (list): List of stock tickers
        weights (list): List of portfolio weights (will be equal if None)
        initial_investment (float): Initial investment amount
        """
        self.tickers = tickers
        self.weights = weights if weights else [1/len(tickers)] * len(tickers)
        self.initial_investment = initial_investment
        
        # Fetch historical data
        self.data = self._fetch_data()
        self.returns = self.data.pct_change()
        
        # Calculate portfolio statistics
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
    def _fetch_data(self, period='2y'):
        """Fetch historical stock data"""
        data = pd.DataFrame()
        for ticker in self.tickers:
            stock = yf.Ticker(ticker)
            data[ticker] = stock.history(period=period)['Close']
        return data
    
    def simulate_portfolio(self, num_simulations=1000, time_horizon=252):
        """
        Run Monte Carlo simulation for portfolio performance
        
        Parameters:
        num_simulations (int): Number of simulations to run
        time_horizon (int): Number of days to simulate (252 = 1 trading year)
        
        Returns:
        array: Matrix of simulation results
        """
        # Initialize simulation array
        simulations = np.zeros((time_horizon, num_simulations))
        
        # Run simulations
        for sim in range(num_simulations):
            # Generate random returns
            rand_returns = np.random.multivariate_normal(
                self.mean_returns,
                self.cov_matrix,
                time_horizon
            )
            
            # Calculate portfolio returns
            portfolio_returns = np.sum(rand_returns * self.weights, axis=1)
            
            # Calculate cumulative returns
            cumulative_returns = np.exp(np.cumsum(portfolio_returns))
            simulations[:, sim] = cumulative_returns * self.initial_investment
            
        return simulations
    
    def analyze_simulations(self, simulations):
        """
        Analyze simulation results and generate statistics
        
        Parameters:
        simulations (array): Matrix of simulation results
        
        Returns:
        dict: Dictionary of statistics
        """
        final_values = simulations[-1, :]
        
        stats_dict = {
            'Initial Investment': f"${self.initial_investment:,.2f}",
            'Mean Final Value': f"${np.mean(final_values):,.2f}",
            'Median Final Value': f"${np.median(final_values):,.2f}",
            'Std Dev of Final Value': f"${np.std(final_values):,.2f}",
            '5th Percentile': f"${np.percentile(final_values, 5):,.2f}",
            '95th Percentile': f"${np.percentile(final_values, 95):,.2f}",
            'Probability of Profit': f"{(final_values > self.initial_investment).mean()*100:.1f}%",
            'Max Final Value': f"${np.max(final_values):,.2f}",
            'Min Final Value': f"${np.min(final_values):,.2f}"
        }
        
        return stats_dict
    
    def plot_simulations(self, simulations):
        """
        Create visualizations of simulation results
        
        Parameters:
        simulations (array): Matrix of simulation results
        """
        plt.figure(figsize=(20, 10))
        
        # Plot 1: Simulation Paths
        plt.subplot(2, 2, 1)
        plt.plot(simulations, alpha=0.1, color='blue')
        plt.plot(simulations.mean(axis=1), color='red', linewidth=2)
        plt.title('Portfolio Value Simulation Paths')
        plt.xlabel('Trading Days')
        plt.ylabel('Portfolio Value ($)')
        
        # Plot 2: Final Values Distribution
        plt.subplot(2, 2, 2)
        sns.histplot(simulations[-1, :], kde=True)
        plt.axvline(self.initial_investment, color='red', linestyle='--')
        plt.title('Distribution of Final Portfolio Values')
        plt.xlabel('Final Portfolio Value ($)')
        plt.ylabel('Frequency')
        
        # Plot 3: Returns QQ Plot
        plt.subplot(2, 2, 3)
        final_returns = (simulations[-1, :] - self.initial_investment) / self.initial_investment
        stats.probplot(final_returns, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Portfolio Returns')
        
        # Plot 4: Portfolio Composition
        plt.subplot(2, 2, 4)
        plt.pie(self.weights, labels=self.tickers, autopct='%1.1f%%')
        plt.title('Portfolio Composition')
        
        plt.tight_layout()
        return plt.gcf()

# Example usage
def run_portfolio_analysis():
    # my portfolio
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    initial_investment = 10000
    
    # Create simulator instance
    simulator = PortfolioSimulator(tickers, weights, initial_investment)
    
    # Run simulations
    simulations = simulator.simulate_portfolio(num_simulations=1000)
    
    # Analyze results
    stats = simulator.analyze_simulations(simulations)
    
    # Print results
    print("\nPortfolio Monte Carlo Simulation Results")
    print("=" * 50)
    print(tabulate([[k, v] for k, v in stats.items()], 
                  headers=['Metric', 'Value'],
                  tablefmt='pretty'))
    
    # Create and show plots
    fig = simulator.plot_simulations(simulations)
    plt.show()

# Run the analysis
run_portfolio_analysis()