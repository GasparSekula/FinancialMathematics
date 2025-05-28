import numpy as np
from scipy.stats import norm


class BSAsianOption:
    def __init__(self, S0, K, T, r, sigma):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
    def _generate_stock_paths(self, num_simulations = 10_000, N = 10):
        dt = self.T / N
        res = np.zeros((num_simulations, N))
        res[:, 0] = self.S0
        for i in range(1, N):
            z = np.random.normal(0, 1, num_simulations)
            res[:, i] = res[:, i-1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * z)
        return res
    
    def estimate_mc_vanilla(self, num_simulations = 100_000, N=10):
        paths = self._generate_stock_paths(num_simulations=num_simulations, N=N)
        
        mean_price_per_path = np.mean(paths, axis = 1)
        payoffs = np.maximum(mean_price_per_path - self.K, 0)
        payoffs = np.exp(-r*T) * payoffs
        cumulative_means = np.cumsum(payoffs) / np.arange(1, num_simulations+1)
        variance = np.var(payoffs, ddof=1) / num_simulations
        
        return cumulative_means, variance
    
    def estimate_mc_control(self, num_simulations = 100_000, N=10):
        paths = self._generate_stock_paths(num_simulations=num_simulations, N=N)
        discount_factor = np.exp(-self.r * self.T) 
        
        payoffs = discount_factor * np.maximum(np.mean(paths, axis=1) - self.K, 0)
        control_variate = discount_factor * np.maximum((np.prod(paths, axis=1)**(1/N) - self.K), 0)
        
        T_i = np.arange(N) * self.T / N
        min_matrix = np.minimum.outer(T_i, T_i)
        alpha = self.sigma * np.sqrt(np.sum(min_matrix) / N**2)
        beta = (self.r - self.sigma**2 / 2) * np.mean(T_i)
        beta = beta + np.log(self.S0)
        
        control_expected = np.exp(beta + alpha**2 / 2) * norm.cdf((beta - np.log(self.K) + alpha**2) / (np.abs(alpha))) - self.K * norm.cdf((beta - np.log(self.K))/(np.abs(alpha)))
        control_expected *= discount_factor
        t = control_expected + payoffs - control_variate
        cumulative_means = np.cumsum(t) / np.arange(1, num_simulations+1)
        variance = np.var(t, ddof=1) / num_simulations
        
        return cumulative_means, variance