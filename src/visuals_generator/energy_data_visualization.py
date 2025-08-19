import matplotlib.pyplot as plt
from src.utils.energy import generate_energy_consumption

# Your parameters (matching your data generation)
e_min, e_max = 20, 220
alpha = 10
p_global_min, p_global_max = 17, 214

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


if __name__ == "__main__":
    processing_times_example = [17, 50, 100, 150, 214]
    plot_energy_distribution(processing_times_example, e_min, e_max, p_global_min, p_global_max, alpha)
