#!/usr/bin/env python3
import os
import itertools
import yaml

# Base configuration dictionary as defined in your YAML.
base_config = {
    "algorithm": "ppo_masked",
    "instances_file": "energy_fjssp/config_4x3x5.pkl",
    "saved_model_name": "4x3x5_rs1",
    "seed": 2,
    "overwrite_split_seed": False,
    "config_description": "4x3x5_rs1 Hyperparameter Test",
    "experiment_save_path": "models",
    "wandb_mode": 2,
    "wandb_project": "jssp_energy",
    "rollout_steps": 2048,
    "gamma": 0.99,
    "n_epochs": 5,              # Fixed value
    "batch_size": 256,          # Fixed value
    "clip_range": 0.2,          # Fixed value
    "ent_coef": 0.015,          # Default, will be overwritten
    "learning_rate": 0.0005,    # Default, will be overwritten
    "policy_layer": [256, 256],
    "policy_activation": "ReLU",
    "value_layer": [256, 256],
    "value_activation": "ReLU",
    "total_instances": 250_000,
    "total_timesteps": 2_000_000_000,
    "train_test_split": 0.9,
    "test_validation_split": 0.8,
    "intermediate_test_interval": 600_000,
    "environment": "energy_env",
    "num_steps_max": 90,
    "log_interval": 200,
    "shuffle": True,
    "reward_strategy": "rs1",
    "reward_scale": 1,
    "test_heuristics": ['rand', 'SPT_FJSSP', 'LEC', 'MTR_EPST', 'LTR_EPST'],
    "success_metric": "makespan_mean"
}

# Define the hyperparameter grid to sweep over.
# Expanded ranges for learning rate and entropy coefficient:
learning_rates = [0.0001, 0.0003, 0.0005, 0.001, 0.005]
ent_coefs = [0.005, 0.01, 0.015, 0.02, 0.05]

# Create the full list of hyperparameter combinations.
all_combinations = list(itertools.product(learning_rates, ent_coefs))

# Create an output directory to store the YAML config files.
output_dir = "limited_hyperparam_configs"
os.makedirs(output_dir, exist_ok=True)

# Generate a YAML file for each configuration.
counter = 0
for lr, ec in all_combinations:
    config = base_config.copy()
    config["learning_rate"] = lr
    config["ent_coef"] = ec

    # Create a unique identifier for this configuration.
    config_suffix = f"lr{lr}_ec{ec}"
    config["config_description"] = config_suffix
    config["saved_model_name"] = config_suffix

    # Define the output file name and write the YAML file.
    filename = f"config_{counter}_{config_suffix}.yaml"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        yaml.dump(config, f)

    print(f"Generated {filepath}")
    counter += 1
