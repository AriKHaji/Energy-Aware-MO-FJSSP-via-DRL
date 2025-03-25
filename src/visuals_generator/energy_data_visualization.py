import numpy as np
import matplotlib.pyplot as plt

# Your parameters (matching your data generation)
e_min, e_max = 20, 220
alpha = 10
p_global_min, p_global_max = 17, 214

def generate_energy_consumption(p, e_min, e_max, p_min, p_max, alpha=4, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    # Normalize processing time clearly
    p_norm = (p - p_min) / (p_max - p_min)

    # Define Beta parameters
    a_param = 1 + alpha * (1 - p_norm)
    b_param = 1 + alpha * p_norm

    # Sample clearly from Beta distribution and scale
    energy = e_min + (e_max - e_min) * np.random.beta(a_param, b_param)

    # Return integer-rounded energy
    return int(np.round(energy))


def plot_energy_distribution(processing_times, e_min, e_max, p_min, p_max, alpha, samples=10000):
    plt.figure(figsize=(10, 6))

    for p in processing_times:
        # Generate many samples clearly for the same processing time
        energy_samples = [
            generate_energy_consumption(p, e_min, e_max, p_min, p_max, alpha)
            for _ in range(samples)
        ]

        # Plot clearly as histogram approximating PDF
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

# p_example = 50
# energy_value = generate_energy_consumption(p_example, 20, 220, 17, 214, alpha=10)
# print(f"Energy consumption: {energy_value}")


processing_times_example = [17, 50, 100, 150, 214]
plot_energy_distribution(processing_times_example, e_min, e_max, p_global_min, p_global_max, alpha)
