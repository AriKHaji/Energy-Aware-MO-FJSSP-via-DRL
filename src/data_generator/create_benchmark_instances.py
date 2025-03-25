import pickle
from pathlib import Path
from typing import List
from rich.console import Console
from rich.table import Table

# Library import
import numpy as np

from src.data_generator.energy_task import EnergyTask

# All Fattahi instances (SFJS)
instance_data = ["""
    2 2 
    2 2 0 25 1 37 2 0 32 1 24
    2 2 0 45 1 65 2 0 21 1 65
    """, #sfjs1 2x2x2
    """
    2 2
    2 1 0 43 2 0 64 1 71
    2 2 0 21 1 35 1 1 43
    """, #sfjs2 2x2x2
    """
    3 2
    2 1 0 43 2 0 87 1 95
    2 2 0 63 1 53 1 1 73
    2 2 0 125 1 135 2 0 43 1 61
    """, #sfjs3 3x2x2
    """
    3 2
    2 2 0 54 1 63 2 0 87 1 95
    2 1 1 120 1 1 152
    2 2 0 125 1 135 2 0 143 1 124
    """, #sfjs4 3x2x2
    """
    3 2
    2 2 0 43 1 36 2 0 64 1 71
    2 2 0 34 1 53 2 0 36 1 21
    2 2 0 21 1 35 2 0 43 1 37
    """, #sfjs5 3x2x2
    """
    3 3
    3 1 0 17 2 0 40 1 130 2 1 50 2 60
    3 1 0 30 2 0 150 1 160 1 2 70
    3 2 0 50 1 60 2 1 170 2 180 2 1 90 2 100
    """, #sfjs6 3x3x3
    """
    3 5
    3 2 0 117 1 125 2 3 140 1 130 2 3 150 4 160
    3 2 0 214 2 150 2 2 55 1 66 2 4 78 2 65
    3 2 0 87 1 62 2 3 70 2 80 2 3 190 4 100
    """, #sfjs7 3x3x5
    """
    3 4
    3 2 0 17 1 25 2 3 40 1 30 2 3 150 2 160
    3 2 0 30 2 50 2 3 55 1 66 2 3 78 2 65
    3 2 0 56 1 62 2 1 70 2 80 2 3 90 2 100
    """, #sfjs8 3x3x4
    """
    3 3
    3 2 0 17 1 25 2 0 40 1 30 2 1 50 2 60
    3 2 0 30 2 50 2 0 50 1 60 2 1 70 2 60
    3 2 0 50 1 60 2 1 70 2 80 2 1 90 2 100
    """, #sfjs9 3x3x3
    """
    4 5
    3 1 0 147 2 3 140 1 130 2 3 150 4 160
    3 2 0 214 2 150 2 2 87 1 66 1 4 178
    3 2 0 87 1 62 1 2 180 2 3 190 4 100
    3 2 0 87 1 65 1 4 173 2 3 145 2 136
    """ #sfjs10
]

### Find global min and max processing times in the dataset
# # Extract all processing times
# processing_times = []
#
# for instance in instance_data:
#     lines = instance.strip().split("\n")[1:]  # skip the first line (job and machine count)
#     for line in lines:
#         parts = list(map(int, line.strip().split()))
#         idx = 1
#         while idx < len(parts):
#             n_alternatives = parts[idx]
#             idx += 1
#             for _ in range(n_alternatives):
#                 machine_idx = parts[idx]
#                 proc_time = parts[idx + 1]
#                 processing_times.append(proc_time)
#                 idx += 2

# print(f"Global minimum processing time: {p_global_min}")
# print(f"Global maximum processing time: {p_global_max}")

def generate_energy_consumption(processing_times, e_min, e_max, alpha=4, random_seed=7):
    """
    Generates stochastic integer energy consumption values inversely related to processing times.

    Parameters:
    - processing_times: list or numpy array of processing times.
    - e_min: minimum possible energy consumption.
    - e_max: maximum possible energy consumption.
    - alpha: shape parameter controlling randomness (default=2).
    - random_seed: seed for reproducibility (default=None).

    Returns:
    - numpy array of integer energy consumption values.
    """
    p_global_min = 17
    p_global_max = 214

    if random_seed is not None:
        np.random.seed(random_seed)

    processing_times = np.array(processing_times)

    # Normalize processing times to [0, 1]
    p_norm = (processing_times - p_global_min) / (p_global_max - p_global_min)

    # Inverse relationship (longer processing -> lower energy)
    a_params = 1 + alpha * (1 - p_norm)
    b_params = 1 + alpha * p_norm

    # Generate stochastic energy values and round to nearest integer
    energy = np.random.beta(a_params, b_params) * (e_max - e_min) + e_min
    energy_int = np.round(energy).astype(int)

    return energy_int

# # Example data
# processing_times = [10, 15, 20, 12]
# e_min, e_max = 5, 20
# alpha = 2
#
# # Generate integer energy consumption data
# energy_values = generate_energy_consumption(processing_times, e_min, e_max, alpha, random_seed=42)
#
# # Print results
# for idx, (p, e) in enumerate(zip(processing_times, energy_values), start=1):
#     print(f"Operation {idx}: Processing Time = {p}, Energy Consumption = {e}")

def parse_fattahi_instance(instance_str: str, e_min: int, e_max: int, alpha=4, random_seed=7) -> List[EnergyTask]:
    """
    Parses an instance from Fattahi et al. and generates EnergyTask instances.

    Parameters:
    - instance_str: Multiline string representing the instance.
    - e_min: Minimum possible energy consumption.
    - e_max: Maximum possible energy consumption.
    - alpha: Shape parameter controlling randomness.

    Returns:
    - List of EnergyTask instances.
    """

    lines = instance_str.strip().split('\n')
    header = lines[0].split()
    n_jobs, n_machines = map(int, header)

    tasks = []
    for job_idx, line in enumerate(lines[1:]):
        parts = list(map(int, line.strip().split()))
        n_operations = parts[0]

        idx = 1
        for task_idx in range(n_operations):
            n_alternatives = parts[idx]
            idx += 1

            machines_mask = [0] * n_machines
            processing_times = {}

            machines = []
            for _ in range(n_alternatives):
                machine_idx = parts[idx]
                proc_time = parts[idx + 1]
                machines.append(machine_idx)
                processing_times[machine_idx] = proc_time
                machines_mask[machine_idx] = 1
                idx += 2

            energy_values = generate_energy_consumption(
                [processing_times[m] for m in machines],
                e_min,
                e_max,
                alpha,
                random_seed
            )

            energy_consumptions = dict(zip(machines, energy_values))

            task = EnergyTask(
                job_index=job_idx,
                task_index=task_idx,
                machines=machines_mask,
                deadline=0,
                done=False,
                _n_machines=len(machines_mask),
                _n_tools=0,
                processing_times=processing_times,
                energy_consumptions=energy_consumptions
            )

            tasks.append(task)

    return tasks

def visualize_instance(instance, num_jobs):
    """
    Visualizes a single instance using a rich table.

    instance: list of Task objects in the instance.
    num_jobs: total number of jobs in this instance.
    """
    # Group tasks by job index.
    jobs = {}
    for task in instance:
        jobs.setdefault(task.job_index, []).append(task)

    console = Console()

    for job_index in range(num_jobs):
        tasks = jobs.get(job_index, [])
        table = Table(title=f"Job {job_index}", show_lines=True)

        table.add_column("Task #", justify="left")
        table.add_column("Machine Mask", justify="center")
        table.add_column("Processing Times", justify="center")
        table.add_column("Energy Consumptions", justify="center")

        for task in sorted(tasks, key=lambda t: t.task_index):
            machine_mask = str(task.machines)
            processing_times = str(task.processing_times)
            energy = str(getattr(task, 'energy_consumptions', {}))
            table.add_row(str(task.task_index), machine_mask, processing_times, energy)

        console.print(table)


# Example usage
if __name__ == "__main__":
    # Get current directory of the script
    current_dir = Path(__file__).parent.resolve()
    print(current_dir)

    data_file_path_parent = Path("../../data/instances/energy_fjssp_benchmark")
    data_file_path_parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    random_seed = 7

    # Generate EnergyTasks from instance
    sfjs01_2x2x2 = parse_fattahi_instance(instance_data[0], e_min=20, e_max=220)
    visualize_instance(sfjs01_2x2x2, 2)
    data = [sfjs01_2x2x2]
    data_file_path = data_file_path_parent / "sfjs01_2x2x2.pkl"
    with open(data_file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("-" * 70)

    sfjs02_2x2x2 = parse_fattahi_instance(instance_data[1], e_min=20, e_max=220)
    visualize_instance(sfjs02_2x2x2, 2)
    data = [sfjs02_2x2x2]
    data_file_path = data_file_path_parent / "sfjs02_2x2x2.pkl"
    with open(data_file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("-" * 70)

    sfjs03_3x2x2 = parse_fattahi_instance(instance_data[2], e_min=20, e_max=220)
    data = [sfjs03_3x2x2]
    data_file_path = data_file_path_parent / "sfjs03_3x2x2.pkl"
    with open(data_file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    visualize_instance(sfjs03_3x2x2, 3)
    print("-" * 70)

    sfjs04_3x2x2 = parse_fattahi_instance(instance_data[3], e_min=20, e_max=220)
    data = [sfjs04_3x2x2]
    data_file_path = data_file_path_parent / "sfjs04_3x2x2.pkl"
    with open(data_file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    visualize_instance(sfjs04_3x2x2, 3)
    print("-" * 70)

    sfjs05_3x2x2 = parse_fattahi_instance(instance_data[4], e_min=20, e_max=220)
    data = [sfjs05_3x2x2]
    data_file_path = data_file_path_parent / "sfjs05_3x2x2.pkl"
    with open(data_file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    visualize_instance(sfjs05_3x2x2, 3)
    print("-" * 70)

    sfjs06_3x3x3 = parse_fattahi_instance(instance_data[5], e_min=20, e_max=220)
    data = [sfjs06_3x3x3]
    data_file_path = data_file_path_parent / "sfjs06_3x3x3.pkl"
    with open(data_file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    visualize_instance(sfjs06_3x3x3, 3)
    print("-" * 70)

    sfjs07_3x3x5 = parse_fattahi_instance(instance_data[6], e_min=20, e_max=220)
    data = [sfjs07_3x3x5]
    data_file_path = data_file_path_parent / "sfjs07_3x3x5.pkl"
    with open(data_file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    visualize_instance(sfjs07_3x3x5, 3)
    print("-" * 70)

    sfjs08_3x3x4 = parse_fattahi_instance(instance_data[7], e_min=20, e_max=220)
    data = [sfjs08_3x3x4]
    data_file_path = data_file_path_parent / "sfjs08_3x3x4.pkl"
    with open(data_file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    visualize_instance(sfjs08_3x3x4, 3)
    print("-" * 70)

    sfjs09_3x3x3 = parse_fattahi_instance(instance_data[8], e_min=20, e_max=220)
    data = [sfjs09_3x3x3]
    data_file_path = data_file_path_parent / "sfjs09_3x3x3.pkl"
    with open(data_file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    visualize_instance(sfjs09_3x3x3, 3)
    print("-" * 70)

    sfjs10_4x3x5 = parse_fattahi_instance(instance_data[9], e_min=20, e_max=220)
    data = [sfjs10_4x3x5]
    data_file_path = data_file_path_parent / "sfjs10_4x3x5.pkl"
    with open(data_file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    visualize_instance(sfjs10_4x3x5, 4)
    print("-" * 70)