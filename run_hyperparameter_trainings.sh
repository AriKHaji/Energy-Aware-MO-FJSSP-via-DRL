#!/bin/bash

CONFIG_FILES=(
    "training/ppo_masked/config_0_lr0.0001_ec0.005.yaml"
    "training/ppo_masked/config_1_lr0.0001_ec0.01.yaml"
    "training/ppo_masked/config_2_lr0.0001_ec0.015.yaml"
    "training/ppo_masked/config_3_lr0.0001_ec0.02.yaml"
    "training/ppo_masked/config_4_lr0.0001_ec0.05.yaml"
    "training/ppo_masked/config_5_lr0.0003_ec0.005.yaml"
    "training/ppo_masked/config_6_lr0.0003_ec0.01.yaml"
    "training/ppo_masked/config_7_lr0.0003_ec0.015.yaml"
    "training/ppo_masked/config_8_lr0.0003_ec0.02.yaml"
    "training/ppo_masked/config_9_lr0.0003_ec0.05.yaml"
    "training/ppo_masked/config_10_lr0.0005_ec0.005.yaml"
    "training/ppo_masked/config_11_lr0.0005_ec0.01.yaml"
    "training/ppo_masked/config_12_lr0.0005_ec0.015.yaml"
    "training/ppo_masked/config_13_lr0.0005_ec0.02.yaml"
    "training/ppo_masked/config_14_lr0.0005_ec0.05.yaml"
    "training/ppo_masked/config_15_lr0.001_ec0.005.yaml"
    "training/ppo_masked/config_16_lr0.001_ec0.01.yaml"
    "training/ppo_masked/config_17_lr0.001_ec0.015.yaml"
    "training/ppo_masked/config_18_lr0.001_ec0.02.yaml"
    "training/ppo_masked/config_19_lr0.001_ec0.05.yaml"
    "training/ppo_masked/config_20_lr0.005_ec0.005.yaml"
    "training/ppo_masked/config_21_lr0.005_ec0.01.yaml"
    "training/ppo_masked/config_22_lr0.005_ec0.015.yaml"
    "training/ppo_masked/config_23_lr0.005_ec0.02.yaml"
    "training/ppo_masked/config_24_lr0.005_ec0.05.yaml"
)

for CONFIG_FILE in "${CONFIG_FILES[@]}"
do
    echo "Starting training with config $CONFIG_FILE at $(date)"
    python -m src.agents.train -fp "$CONFIG_FILE"
done

echo "All training runs completed at $(date)!"
