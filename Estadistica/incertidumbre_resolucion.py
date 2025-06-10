import numpy as np
import matplotlib.pyplot as plt


def montecarlo_gaussian_estimation(sigma, delta, n, num_simulations=1000):
    np.random.seed(0)  # For reproducibility
    errors = []
    mean_values = np.linspace(-3*sigma*delta, 3*sigma*delta, num_simulations)

    for mean in mean_values:
        # Simulate n measurements from a Gaussian distribution
        samples = np.random.normal(loc=mean, scale=sigma, size=n)

        # Quantize the measurements based on the instrument resolution
        quantized_samples = np.round(samples / delta) * delta

        # Estimate the mean from quantized samples
        estimated_mean = np.mean(quantized_samples)

        # Calculate the estimation error
        error = np.abs(estimated_mean - mean)
        errors.append(error)

    return np.array(errors)


def delta_sweep(deltas, sigma, n, n_simulations):
    rmse = np.zeros(len(deltas))
    for kd, delta in enumerate(deltas):
        errors = montecarlo_gaussian_estimation(sigma, delta, n, n_simulations)
        rmse[kd] = np.sqrt(np.mean(errors**2))
    return rmse


# Example usage
sigma = 1.0
n_deltas = 40
deltas = np.logspace(-1, 2, n_deltas) * sigma
n = 10
n_simulations = 5000

rmse = delta_sweep(deltas, sigma, n, n_simulations)

uniform_uncertainty_single = deltas / np.sqrt(12)
uniform_uncertainty_multiple = uniform_uncertainty_single / np.sqrt(n)
std_mean_uncertainty = sigma / np.sqrt(n) * np.ones(n_deltas)
combined_uncertainty_usingle = np.sqrt(std_mean_uncertainty**2 + uniform_uncertainty_single**2)
combined_uncertainty_multiple = np.sqrt(std_mean_uncertainty**2 + uniform_uncertainty_multiple**2)
plt.semilogx(deltas, rmse, label='RMSE Monte Carlo')
plt.semilogx(deltas, uniform_uncertainty_single, label='Uniform Uncertainty (Single)', linestyle='--')
plt.semilogx(deltas, uniform_uncertainty_multiple, label='Uniform Uncertainty (Multiple)', linestyle='-.')
plt.semilogx(deltas, std_mean_uncertainty, label='Standard Deviation of Mean', linestyle='--')
plt.semilogx(deltas, combined_uncertainty_usingle, label='Combined Uncertainty (Uniform Single)', linestyle='--')
plt.semilogx(deltas, combined_uncertainty_multiple, label='Combined Uncertainty (Uniform Multiple)', linestyle='-.')
plt.xlabel('Instrument Resolution / Sigma')
plt.ylabel('- Estimation Error (RMSE), -- Standard Deviation')
plt.title(f'Monte Carlo Simulation of Mean Estimation Error ({n_simulations} simulations, {n} measurements)')
plt.legend()
plt.grid(True)
for line in plt.gca().get_lines():
    line.set_linewidth(2)
plt.show()

ns = [10, 15, 30]
fig, axs = plt.subplots(len(ns), 1)
for kn, n in enumerate(ns):
    uniform_uncertainty_multiple = uniform_uncertainty_single / np.sqrt(n)
    std_mean_uncertainty = sigma / np.sqrt(n) * np.ones(n_deltas)
    combined_uncertainty_usingle = np.sqrt(std_mean_uncertainty**2 + uniform_uncertainty_single**2)
    combined_uncertainty_multiple = np.sqrt(std_mean_uncertainty**2 + uniform_uncertainty_multiple**2)
    
    rmse = delta_sweep(deltas, sigma, n, n_simulations)
    ratio_combined_single = combined_uncertainty_usingle / rmse
    ratio_combined_multiple = combined_uncertainty_multiple / rmse

    axs[kn].semilogx(deltas, ratio_combined_single, label='Combined Uncertainty (Uniform Single) / RMSE', linewidth=2)
    axs[kn].semilogx(deltas, ratio_combined_multiple, label='Combined Uncertainty (Uniform Multiple) / RMSE', linewidth=2)    
    axs[kn].grid(True)
    axs[kn].set_ylabel('Ratio')
    # axs[kn].set_title(f'n = {n}', loc='left', pad=-100)
    axs[kn].text(0.01, 0.98, f'n = {n}', transform=axs[kn].transAxes, fontsize=12, va='top', ha='left')
plt.xlabel('Instrument Resolution / Sigma')
plt.suptitle('Ratio of Combined Uncertainty to RMSE')
axs[0].legend()
plt.show()
