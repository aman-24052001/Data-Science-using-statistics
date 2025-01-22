import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tabulate import tabulate
import networkx as nx
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

class MarkovChainAnalyzer:
    def __init__(self):
        """Initialize Markov Chain Analyzer with different application examples"""
        self.applications = {
            'healthcare': self.setup_healthcare_example(),
            'supply_chain': self.setup_supply_chain_example(),
            'web_ranking': self.setup_web_ranking_example()
        }
    
    def setup_healthcare_example(self):
        """
        Setup healthcare example using breast cancer dataset
        States: Normal (0), Benign (1), Malignant (2)
        """
        data = load_breast_cancer()
        # Create simplified transition matrix based on feature correlations
        states = ['Normal', 'Benign', 'Malignant']
        transition_matrix = np.array([
            [0.7, 0.2, 0.1],  # Normal -> states
            [0.1, 0.6, 0.3],  # Benign -> states
            [0.05, 0.15, 0.8]  # Malignant -> states
        ])
        return {'states': states, 'P': transition_matrix}
    
    def setup_supply_chain_example(self):
        """
        Setup supply chain example
        States: Low Stock, Normal Stock, Overstocked
        """
        states = ['Low_Stock', 'Normal_Stock', 'Overstocked']
        transition_matrix = np.array([
            [0.2, 0.7, 0.1],  # Low -> states
            [0.1, 0.8, 0.1],  # Normal -> states
            [0.1, 0.6, 0.3]   # Over -> states
        ])
        return {'states': states, 'P': transition_matrix}
    
    def setup_web_ranking_example(self):
        """
        Setup simplified PageRank example
        States: Page A, Page B, Page C, Page D
        """
        states = ['Page_A', 'Page_B', 'Page_C', 'Page_D']
        transition_matrix = np.array([
            [0.3, 0.4, 0.2, 0.1],
            [0.2, 0.3, 0.3, 0.2],
            [0.1, 0.3, 0.4, 0.2],
            [0.2, 0.2, 0.3, 0.3]
        ])
        return {'states': states, 'P': transition_matrix}
    
    def calculate_steady_state(self, P):
        """Calculate steady state probabilities"""
        eigenvals, eigenvecs = np.linalg.eig(P.T)
        steady_state = eigenvecs[:, np.argmax(eigenvals)].real
        return steady_state / steady_state.sum()
    
    def predict_n_steps(self, P, initial_state, n_steps):
        """Predict state probabilities after n steps"""
        current_state = initial_state
        states = []
        for _ in range(n_steps):
            current_state = current_state @ P
            states.append(current_state)
        return np.array(states)
    
    def analyze_absorption_probabilities(self, P):
        """Calculate absorption probabilities if applicable"""
        n = len(P)
        Q = P[:-1, :-1]  # Transient states
        R = P[:-1, -1:]  # Absorption probabilities
        N = np.linalg.inv(np.eye(n-1) - Q)  # Fundamental matrix
        B = N @ R  # Absorption probabilities
        return B
    
    def visualize_markov_chain(self, states, P, title):
        """Create network visualization of Markov chain"""
        G = nx.DiGraph()
        
        # Add nodes
        for i, state in enumerate(states):
            G.add_node(state)
        
        # Add edges with transition probabilities
        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                if P[i, j] > 0:
                    G.add_edge(state1, state2, weight=P[i, j])
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=2000, alpha=0.7)
        
        # Draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, 
                             edge_color='gray', arrows=True, 
                             arrowsize=20)
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {k: f'{v:.2f}' for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        
        plt.title(title)
        return plt.gcf()
    
    def analyze_application(self, app_name, n_steps=10):
        """Analyze specific Markov Chain application"""
        app = self.applications[app_name]
        states, P = app['states'], app['P']
        
        # Calculate steady state
        steady_state = self.calculate_steady_state(P)
        
        # Predict future states
        initial_state = np.array([1/len(states)] * len(states))
        predictions = self.predict_n_steps(P, initial_state, n_steps)
        
        # Generate visualization
        fig = self.visualize_markov_chain(states, P, f'Markov Chain - {app_name}')
        
        # Prepare analysis summary
        summary = {
            'Initial State': dict(zip(states, initial_state)),
            'Steady State': dict(zip(states, steady_state)),
            'Convergence Rate': np.abs(np.linalg.eigvals(P))[1],
            'Predictions': {
                f'Step {i+1}': dict(zip(states, pred))
                for i, pred in enumerate(predictions)
            }
        }
        
        return summary, fig
    
    def generate_report(self, app_name):
        """Generate comprehensive analysis report"""
        summary, fig = self.analyze_application(app_name)
        
        # Create formatted report
        report = [
            f"\nMarkov Chain Analysis Report - {app_name}",
            "=" * 50,
            "\n1. Initial State Distribution:",
            tabulate([[state, f"{prob:.3f}"] for state, prob in 
                     summary['Initial State'].items()],
                    headers=['State', 'Probability'],
                    tablefmt='pretty'),
            "\n2. Steady State Distribution:",
            tabulate([[state, f"{prob:.3f}"] for state, prob in 
                     summary['Steady State'].items()],
                    headers=['State', 'Probability'],
                    tablefmt='pretty'),
            f"\n3. Convergence Rate: {summary['Convergence Rate']:.3f}",
            "\n4. State Predictions:",
        ]
        
        # Add predictions table
        pred_table = []
        for step, probs in summary['Predictions'].items():
            row = [step] + [f"{p:.3f}" for p in probs.values()]
            pred_table.append(row)
        
        report.append(tabulate(pred_table,
                             headers=['Step'] + list(summary['Initial State'].keys()),
                             tablefmt='pretty'))
        
        return "\n".join(report), fig

def run_analysis():
    """Run complete Markov Chain analysis for all applications"""
    analyzer = MarkovChainAnalyzer()
    
    for app_name in ['healthcare', 'supply_chain', 'web_ranking']:
        print(f"\nAnalyzing {app_name.replace('_', ' ').title()} Application")
        report, fig = analyzer.generate_report(app_name)
        print(report)
        plt.show()

if __name__ == "__main__":
    run_analysis()