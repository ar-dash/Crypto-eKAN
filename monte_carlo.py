import numpy as np

# === GBM-Based Monte Carlo Prediction ===
def monte_carlo_next_day_gbm(prices, num_simulations=10000):
    log_returns = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)
    dt = 1 / 365

    Z = np.random.normal(0, 1, num_simulations)
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    simulated_returns = drift + diffusion

    predicted_prices = prices[-1] * np.exp(simulated_returns)
    return np.mean(predicted_prices)

# === CIR Model for mean-reverting behavior ===
def monte_carlo_next_day_cir(prices, num_simulations=10000):
    log_returns = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)
    dt = 1 / 365

    Z = np.random.normal(0, 1, num_simulations)
    simulated_returns = 1 * (mu - max(0, prices[-1])) * dt + Z * sigma * (max(0, prices[-1]) * dt) * 0.5
    predicted_prices = prices[-1] * np.exp(simulated_returns)
    return np.mean(predicted_prices)

# === Normal Distribution Based Monte Carlo ===
def monte_carlo_next_day_nor(prices, num_simulations=10000):
    log_returns = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)

    Z = np.random.normal(mu, sigma, num_simulations)
    predicted_prices = prices[-1] * np.exp(Z)
    return np.mean(predicted_prices)

# === Rolling Monte Carlo Window (GBM) ===
def run_rolling_predictions(prices, model_func, window_size=4, num_simulations=1000):
    predictions, actuals = [], []
    for i in range(window_size, len(prices)-1):
        window = prices[i-window_size:i]
        actual = prices[i]
        predicted = model_func(window, num_simulations)
        predictions.append(predicted)
        actuals.append(actual)
    return np.array(predictions), np.array(actuals)
