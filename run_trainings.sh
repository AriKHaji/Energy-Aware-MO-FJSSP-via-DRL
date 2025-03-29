#!/bin/bash

# List all your config files explicitly
CONFIG_FILES=(
    "training/ppo_masked/config_efjssp_4x3x5.yaml"
    "training/ppo_masked/config_efjssp_3x3x5.yaml"
    "training/ppo_masked/config_efjssp_3x3x4.yaml"
    "training/ppo_masked/config_efjssp_3x3x3.yaml"
    "training/ppo_masked/config_efjssp_3x2x2.yaml"
    "training/ppo_masked/config_efjssp_2x2x2.yaml"
)

for CONFIG_FILE in "${CONFIG_FILES[@]}"
do
    echo "Starting training with config $CONFIG_FILE at $(date)"
    python -m src.agents.train -fp "$CONFIG_FILE"
done

echo "All training runs completed at $(date)!"