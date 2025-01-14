import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import Union, List, Tuple, Callable
from dataclasses import dataclass

@dataclass
class BootstrapResult:
    """Class to store bootstrap analysis results"""
    statistic_name: str
    original_value: float
    bootstrap_values: np.ndarray
    confidence_interval: Tuple[float, float]
    
class BootstrapAnalyzer:
    """Class for performing bootstrap analysis on datasets"""
    
    def __init__(self, data: pd.DataFrame, n_iterations: int = 1000, random_state: int = 42):
        """
        Initialize the bootstrap analyzer
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        n_iterations : int
            Number of bootstrap iterations
        random_state : int
            Random seed for reproducibility
        """
        self.data = data
        self.n_iterations = n_iterations
        self.random_state = random_state
        np.random.seed(random_state)
        
    def _bootstrap_sample(self, data: np.ndarray) -> np.ndarray:
        """Generate a bootstrap sample"""
        indices = np.random.randint(0, len(data), size=len(data))
        return data[indices]
    
    def _calculate_statistic(self, data: np.ndarray, statistic: Union[str, Callable]) -> float:
        """Calculate the specified statistic"""
        if isinstance(statistic, str):
            if statistic == 'mean':
                return np.mean(data)
            elif statistic == 'median':
                return np.median(data)
            elif statistic == 'std':
                return np.std(data)
        else:
            return statistic(data)
    
    def get_confidence_interval(
        self, 
        column: str, 
        statistic: Union[str, Callable] = 'mean',
        confidence_level: float = 0.95
    ) -> BootstrapResult:
        """
        Calculate confidence interval using bootstrap
        
        Parameters:
        -----------
        column : str
            Column name to analyze
        statistic : str or callable
            Statistic to compute
        confidence_level : float
            Confidence level (0-1)
        
        Returns:
        --------
        BootstrapResult
            Results containing original statistic, bootstrap values, and confidence interval
        """
        data = self.data[column].values
        original_value = self._calculate_statistic(data, statistic)
        bootstrap_values = np.zeros(self.n_iterations)
        
        for i in range(self.n_iterations):
            sample = self._bootstrap_sample(data)
            bootstrap_values[i] = self._calculate_statistic(sample, statistic)
            
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_values, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_values, (1 - alpha/2) * 100)
        
        return BootstrapResult(
            statistic_name=statistic if isinstance(statistic, str) else statistic.__name__,
            original_value=original_value,
            bootstrap_values=bootstrap_values,
            confidence_interval=(ci_lower, ci_upper)
        )
    
    def hypothesis_test(
        self,
        column: str,
        group_column: str,
        group1: str,
        group2: str,
        statistic: Union[str, Callable] = 'mean',
        alternative: str = 'two-sided'
    ) -> Tuple[float, float]:
        """
        Perform bootstrap hypothesis test
        
        Parameters:
        -----------
        column : str
            Column name for analysis
        group_column : str
            Column name containing group labels
        group1, group2 : str
            Group labels to compare
        statistic : str or callable
            Statistic to compare
        alternative : str
            Type of test ('two-sided', 'greater', 'less')
            
        Returns:
        --------
        Tuple[float, float]
            Effect size and p-value
        """
        data1 = self.data[self.data[group_column] == group1][column].values
        data2 = self.data[self.data[group_column] == group2][column].values
        
        # Calculate observed difference
        observed_diff = (self._calculate_statistic(data1, statistic) - 
                       self._calculate_statistic(data2, statistic))
        
        # Bootstrap differences
        n_boot_diff = np.zeros(self.n_iterations)
        for i in range(self.n_iterations):
            boot1 = self._bootstrap_sample(data1)
            boot2 = self._bootstrap_sample(data2)
            n_boot_diff[i] = (self._calculate_statistic(boot1, statistic) - 
                            self._calculate_statistic(boot2, statistic))
            
        # Calculate p-value based on alternative hypothesis
        if alternative == 'two-sided':
            p_value = np.mean(np.abs(n_boot_diff) >= np.abs(observed_diff))
        elif alternative == 'greater':
            p_value = np.mean(n_boot_diff >= observed_diff)
        else:  # 'less'
            p_value = np.mean(n_boot_diff <= observed_diff)
            
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
        effect_size = observed_diff / pooled_std
        
        return effect_size, p_value
    
    def plot_bootstrap_distribution(self, result: BootstrapResult, figsize: Tuple[int, int] = (10, 6)):
        """Plot bootstrap distribution with confidence interval"""
        plt.figure(figsize=figsize)
        sns.histplot(result.bootstrap_values, kde=True)
        plt.axvline(result.original_value, color='red', linestyle='--', label='Original Value')
        plt.axvline(result.confidence_interval[0], color='green', linestyle='--', label='CI Lower')
        plt.axvline(result.confidence_interval[1], color='green', linestyle='--', label='CI Upper')
        plt.title(f'Bootstrap Distribution of {result.statistic_name}')
        plt.xlabel(f'{result.statistic_name} Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

def main():
    """Example usage with Titanic dataset"""
    # Load Titanic dataset
    titanic = sns.load_dataset('titanic')
    
    # Initialize analyzer
    analyzer = BootstrapAnalyzer(titanic)
    
    # Calculate confidence interval for fare
    fare_result = analyzer.get_confidence_interval('fare', statistic='mean')
    print("\nFare Analysis:")
    print(f"Mean fare: ${fare_result.original_value:.2f}")
    print(f"95% CI: (${fare_result.confidence_interval[0]:.2f}, ${fare_result.confidence_interval[1]:.2f})")
    
    # Plot bootstrap distribution
    analyzer.plot_bootstrap_distribution(fare_result)
    
    # Perform hypothesis test for fare difference between classes
    effect_size, p_value = analyzer.hypothesis_test(
        'fare', 'class', 'First', 'Third', statistic='mean'
    )
    print("\nHypothesis Test Results:")
    print(f"Effect size (Cohen's d): {effect_size:.2f}")
    print(f"P-value: {p_value:.4f}")

if __name__ == "__main__":
    main()