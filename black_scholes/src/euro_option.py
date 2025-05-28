import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

def estimate_mc_classic(S0, T, K, sigma, r, num_samples):
    z = np.random.normal(0, 1, num_samples)
    ST = S0 * np.exp((sigma * np.sqrt(T) * z) + ((r - sigma**2/2)*T))
    discount_factor = np.exp(-r * T)
    payoff = np.maximum(ST - K, 0)
    mean_price = np.mean(payoff) * discount_factor
    variance = np.var(payoff, ddof=1) * discount_factor**2 / num_samples
    return mean_price, variance
    
    
def estimate_mc_control_variates(S0, T, K, sigma, r, num_samples):
    z = np.random.normal(0, 1, num_samples)
    ST = S0 * np.exp((sigma * np.sqrt(T) * z) + ((r - sigma**2/2)*T))
    discount_factor = np.exp(-r * T)
    payoff = discount_factor * np.maximum(ST - K, 0)
    
    mu_k = S0
    
    control_variate = discount_factor * ST
    t = payoff - control_variate 
    mean_price = np.mean(t)
    variance = np.var(t, ddof=1) * discount_factor**2 / num_samples
    res = mu_k + mean_price
    return res, variance


def estimate_mc_control_turbo(S0, T, K, sigma, r, num_samples):
    z = np.random.normal(0, 1, num_samples)
    ST = S0 * np.exp((sigma * np.sqrt(T) * z) + ((r - sigma**2/2)*T))
    discount_factor = np.exp(-r * T)
    payoff = np.maximum(ST - K, 0)
    
    X = discount_factor * payoff
    Y = ST
    b_star = np.cov(X,Y)[0,1] / np.var(Y)
    
    mu_k = b_star * np.mean(ST)
    t = X - b_star * ST 
    mean_price = np.mean(t)
    res = mean_price + mu_k
    variance = np.var(t, ddof=1) * discount_factor ** 2 / num_samples
    return res, variance

def estimate_mc_antithetic(S0, T, K, sigma, r, num_samples):
    z = np.random.normal(0, 1, num_samples // 2)
    z_antithetic = -z
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    ST_antithetic = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z_antithetic)
    
    payoff = np.maximum(ST - K, 0)
    payoff_antithetic = np.maximum(ST_antithetic - K, 0)
    combined_payoff = 0.5 * (payoff + payoff_antithetic)
    
    discount_factor = np.exp(-r * T)
    
    mean_price = np.mean(combined_payoff) * discount_factor
    variance = np.var(combined_payoff, ddof=1) * discount_factor**2 / num_samples
    
    return mean_price, variance


def estimate_mc_stratified(S0, T, K, sigma, r, num_samples, k = 100):
    n_in_stratum = num_samples // k
    prices = []
    
    for i in range(k):
        u = np.random.uniform(0, 1, n_in_stratum)
        z_stratum_i = norm.ppf((i+u)/k)
        ST_stratum_i = S0 * np.exp((sigma * np.sqrt(T) * z_stratum_i) + ((r - sigma**2/2)*T))
        payoff = np.maximum(ST_stratum_i - K, 0)
        prices.append(np.mean(payoff))
        
    discount_factor = np.exp(-r * T)
    res = np.mean(prices) * discount_factor 
    var = np.var(prices, ddof=1) * discount_factor**2 / k / num_samples
    return res, var

def estimate_mc_importance_sampling(S0, T, K, sigma, r, num_samples, mu=2):
    y = np.random.normal(mu, T, num_samples)
    pdf_quotient = np.exp((-2*mu*y + mu**2) / (2*T))
    ST = S0 * np.exp(sigma * y + (r - sigma**2/2)*T)
    payoff = np.maximum(ST - K, 0)
    discount_factor = np.exp(-r * T)
    
    t = payoff * pdf_quotient
    
    mean_price = discount_factor * np.mean(t)
    variance = np.var(t, ddof=1) * discount_factor**2 / num_samples
    
    return mean_price, variance