import numpy as np
import matplotlib.pyplot as plt

from src.euro_option import estimate_mc_importance_sampling


def plot_mc_convergence(estimator, S0, T, K, sigma, r, num_samples, method_name, theroretical_price=None, N=100, start_plot=1):
    sample_sizes = np.linspace(start_plot, num_samples, N, dtype=int)
    estimates = []
    variances = []
    for n in sample_sizes:
        res, var = estimator(S0, T, K, sigma, r, n)
        estimates.append(res)
        variances.append(var)
    
    option_type = "In-the-money" if S0 > K else "Out-of-the-money"
    
    print(f"Model: {method_name}")
    print(f"Option type: {option_type}")
    print(f"Parameters: S0={S0}, K={K}, sigma={sigma}, r={r}, T={T}, num_samples={num_samples}")
    if theroretical_price is not None:
        print(f"Theoretical value: {theroretical_price:.4f}")
    print(f"Estimated value: {estimates[-1]:.4f}")
    print(f"Variance: {variances[-1]:.4f}")
    print("-" * 50)
    
    plt.figure(figsize=(12, 5))
    plt.plot(sample_sizes, np.array(estimates), label=f'{method_name} Estimate', color='blue')
    if theroretical_price is not None:
        plt.axhline(y=theroretical_price, color='r', linestyle='--', label='Theoretical Price')
    plt.xlabel('Number of Samples')
    plt.ylabel('Estimated Price')
    plt.title(f'{method_name} Price Convergence')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 5))
    plt.plot(sample_sizes, variances, label=f'{method_name} Variance', color='blue')
    plt.xlabel('Number of Samples')
    plt.ylabel('Variance')
    plt.title(f'{method_name} Variance Convergence')
    plt.legend()
    plt.show()
    

def plot_importance_sampling_variance(mus_list, S0, T, K, sigma, r, start_plot = 1000, num_samples=100_000, N = 100, avg_last=10):
    sample_sizes = np.linspace(start_plot, num_samples, N, dtype=int)
    variances_dict = {mu: [] for mu in mus_list}
    for mu in mus_list:
        for n in sample_sizes:
            _, var = estimate_mc_importance_sampling(S0, T, K, sigma, r, n, mu=mu)
            variances_dict[mu].append(var)

    option_type = "In-the-money" if S0 > K else "Out-of-the-money"

    plt.figure(figsize=(12, 6))
    for mu, vars_ in variances_dict.items():
        plt.plot(sample_sizes, vars_, label=f'mu={mu}')
    plt.xlabel('Number of Samples')
    plt.ylabel('Variance')
    plt.title(f'Variance Convergence of Importance Sampling vs mu ({option_type})')
    plt.grid(True)
    plt.legend(title='mu')
    plt.xscale('log')
    plt.show()

    print(f'Average variances in final {avg_last} numbers of samples')
    for mu, vars_ in variances_dict.items():
        avg = np.mean(vars_[-avg_last:])
        print(f"mu={mu}:  var = {avg:.8f}")
        

