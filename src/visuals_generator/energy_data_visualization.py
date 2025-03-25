import numpy as np
import matplotlib.pyplot as plt

# Your parameters (matching your data generation)
e_min, e_max = 20, 220
alpha = 10
p_global_min, p_global_max = 17, 214

# Use your existing function directly
def generate_energy_consumption(processing_times, e_min, e_max, alpha=4, random_seed=7):
    if random_seed is not None:
        np.random.seed(random_seed)

    processing_times = np.array(processing_times)

    p_norm = (processing_times - p_global_min) / (p_global_max - p_global_min)

    a_params = 1 + alpha * (1 - p_norm)
    b_params = 1 + alpha * p_norm

    energy = np.random.beta(a_params, b_params) * (e_max - e_min) + e_min
    energy_int = np.round(energy).astype(int)

    return energy_int

def plot_energy_distribution(processing_times, e_min, e_max, alpha, samples=10000):
    plt.figure(figsize=(10, 6))

    for p in processing_times:
        # Generate many samples for a single processing time to approximate distribution
        energy_samples = generate_energy_consumption(
            [p]*samples, e_min, e_max, alpha
        )

        # Plot histogram as PDF approximation
        plt.hist(
            energy_samples,
            bins=100,
            density=True,
            alpha=0.6,
            label=f'Processing time = {p}'
        )

    plt.title('Energy Consumption Distributions for Different Processing Times')
    plt.xlabel('Energy Consumption')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example processing times (lowest, mid-low, mid, mid-high, highest)
processing_times_example = [17, 50, 100, 150, 214]

# Plot
plot_energy_distribution(processing_times_example, e_min, e_max, alpha)
