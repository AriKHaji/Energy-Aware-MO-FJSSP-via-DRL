#!/usr/bin/env python3
import os
import yaml
import re
from collections import defaultdict

# Benchmark files
benchmark_instances = [
    "sfjs01_2x2x2.pkl",
    "sfjs02_2x2x2.pkl",
    "sfjs03_3x2x2.pkl",
    "sfjs04_3x2x2.pkl",
    "sfjs05_3x2x2.pkl",
    "sfjs06_3x3x3.pkl",
    "sfjs07_3x3x5.pkl",
    "sfjs08_3x3x4.pkl",
    "sfjs09_3x3x3.pkl",
    "sfjs10_4x3x5.pkl"
]

# Group benchmarks by instance size
benchmarks_by_instance = defaultdict(list)
for filename in benchmark_instances:
    match = re.search(r"_(\d+x\d+x\d+)\.pkl", filename)
    if match:
        instance_size = match.group(1)
        benchmarks_by_instance[instance_size].append(filename)

# Instance sizes and reward strategies
instance_sizes = ["2x2x2", "3x2x2", "3x3x3", "3x3x4", "3x3x5", "4x3x5"]
reward_strategies = [f"rs{i}" for i in range(1, 9)]

# Output directory
output_dir = "../../config/testing/ppo_masked"
os.makedirs(output_dir, exist_ok=True)

# Create config files
for instance_size in instance_sizes:
    for benchmark_file in benchmarks_by_instance.get(instance_size, []):
        benchmark_name = benchmark_file.replace(".pkl", "")
        benchmark_prefix = benchmark_name.split("_")[0]  # e.g. 'sfjs01'
        full_heuristics_written = False  # Track whether heuristics were added for this benchmark

        for rs in reward_strategies:
            model_name = f"{instance_size}_{rs}"
            description = f"{benchmark_prefix}_{model_name}"

            test_config = {
                "algorithm": "ppo_masked",
                "instances_file": f"energy_fjssp_benchmark/{benchmark_file}",
                "environment": "energy_env",
                "reward_strategy": "dense_makespan_reward",
                "reward_scale": 1,
                "saved_model_name": model_name,
                "seed": 7,
                "config_description": description,
                "experiment_save_path": "models",
                "wandb_mode": 2,
                "wandb_project": "jssp_energy",
                "test_heuristics": ['rand', 'SPT_FJSSP', 'LEC', 'MTR_EPST', 'LTR_EPST'] if not full_heuristics_written else []
            }

            full_heuristics_written = True  # After the first config for this benchmark

            # Save file as <benchmark>_<model>.yaml
            config_filename = f"{benchmark_prefix}_{model_name}.yaml"
            config_path = os.path.join(output_dir, config_filename)

            with open(config_path, "w") as f:
                yaml.dump(test_config, f)

            print(f"Generated {config_path}")
