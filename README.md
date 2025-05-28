# 💸 Financial Mathematics

A comprehensive repository dedicated to implementing core financial mathematics models and simulation techniques. This project covers stochastic processes, option pricing models (CRR and Black-Scholes), Monte Carlo methods, and advanced variance reduction techniques — all from scratch without relying on built-in random generators or models.

---

## 🛠️ Technologies Used

- **NumPy**: Efficient numerical computations
- **Pandas**: Data handling and manipulation
- **SciPy**: Scientific computations
- **Cython**: Speeding up Python code

---

## 📁 Project Structure

| Directory | Topic | Highlights |
|----------|-------|------------|
| `rng` | Pseudorandom Number Generators | Implementation of various PRNGs from scratch, built on linear PRNG logic. No built-in generators used. |
| `stochastic_processes` | Stochastic Process Simulations | Generate and study trajectories of key processes like Poisson and Wiener. |
| `crr_model` | Cox-Ross-Rubinstein Model | Implements CRR binomial model with support for multiple payout options: European Call, European Put, and max-value-at-path. |
| `crr_mc` | Monte Carlo for CRR Model | Uses Monte Carlo estimation for option pricing within the CRR framework. |
| `variance_reduction` | Variance Reduction Techniques | Implements stratified sampling, importance sampling, control variates, and other advanced techniques to improve estimation efficiency. |
| `black_scholes` | Black-Scholes Option Pricing | Option pricing using various Monte Carlo and variance reduction strategies. Supports European, Asian, and complex exotic options. |

---


