#!/usr/bin/env python3
import os
import yaml

# List of instance sizes and reward strategies
instance_sizes = ["2x2x2", "3x2x2", "3x3x3", "3x3x4", "3x3x5", "4x3x5"]
reward_strategies = [f"rs{i}" for i in range(1, 9)]

# Output directory
output_dir = "../../config/training/ppo_masked"
os.makedirs(output_dir, exist_ok=True)

# Loop through all instance size and reward strategy combinations
for instance in instance_sizes:
    for rs_tag in reward_strategies:
        config = {
            "algorithm": "ppo_masked",
            "instances_file": f"energy_fjssp/config_{instance}.pkl",
            "saved_model_name": f"{instance}_{rs_tag}",
            "seed": 2,
            "overwrite_split_seed": False,
            "config_description": f"{instance}_{rs_tag}",
            "experiment_save_path": "models",
            "wandb_mode": 2,
            "wandb_project": "jssp_energy",
            "rollout_steps": 2048,
            "gamma": 0.99,
            "n_epochs": 5,
            "batch_size": 256,
            "clip_range": 0.2,
            "ent_coef": 0.015,
            "learning_rate": 0.0005,
            "policy_layer": [256, 256],
            "policy_activation": "ReLU",
            "value_layer": [256, 256],
            "value_activation": "ReLU",
            "total_instances": 1_000_000,
            "total_timesteps": 2_000_000_000,
            "train_test_split": 0.9,
            "test_validation_split": 0.8,
            "intermediate_test_interval": 600_000,
            "environment": "energy_env",
            "num_steps_max": 90,
            "log_interval": 200,
            "shuffle": True,
            "reward_strategy": rs_tag,
            "reward_scale": 1,
            "test_heuristics": ['rand', 'SPT_FJSSP', 'LEC', 'MTR_EPST', 'LTR_EPST'],
            "success_metric": "makespan_mean"
        }

        filename = f"{rs_tag}_{instance}.yaml"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            yaml.dump(config, f)

        print(f"Generated {filepath}")