#!/bin/bash

# List all your config files explicitly
CONFIG_FILES=(
#    "training/ppo_masked/rs1_2x2x2.yaml"
#    "training/ppo_masked/rs1_3x2x2.yaml"
#    "training/ppo_masked/rs1_3x3x3.yaml"
#    "training/ppo_masked/rs1_3x3x4.yaml"
#    "training/ppo_masked/rs1_3x3x5.yaml"
#    "training/ppo_masked/rs1_4x3x5.yaml"
#    "training/ppo_masked/rs2_2x2x2.yaml"
#    "training/ppo_masked/rs2_3x2x2.yaml"
#    "training/ppo_masked/rs2_3x3x3.yaml"
#    "training/ppo_masked/rs2_3x3x4.yaml"
#    "training/ppo_masked/rs2_3x3x5.yaml"
#    "training/ppo_masked/rs2_4x3x5.yaml"
#    "training/ppo_masked/rs3_2x2x2.yaml"
#    "training/ppo_masked/rs3_3x2x2.yaml"
#    "training/ppo_masked/rs3_3x3x3.yaml"
#    "training/ppo_masked/rs3_3x3x4.yaml"
#    "training/ppo_masked/rs3_3x3x5.yaml"
#    "training/ppo_masked/rs3_4x3x5.yaml"

#    "training/ppo_masked/rs5_2x2x2.yaml"
#    "training/ppo_masked/rs5_3x2x2.yaml"
#    "training/ppo_masked/rs5_3x3x3.yaml"
#    "training/ppo_masked/rs5_3x3x4.yaml"
#    "training/ppo_masked/rs5_3x3x5.yaml"
#    "training/ppo_masked/rs5_4x3x5.yaml"

    "training/ppo_masked/rs7_2x2x2.yaml"
    "training/ppo_masked/rs7_3x2x2.yaml"
    "training/ppo_masked/rs7_3x3x3.yaml"
    "training/ppo_masked/rs7_3x3x4.yaml"
    "training/ppo_masked/rs7_3x3x5.yaml"
    "training/ppo_masked/rs7_4x3x5.yaml"

    "training/ppo_masked/rs6_2x2x2.yaml"
    "training/ppo_masked/rs6_3x2x2.yaml"
    "training/ppo_masked/rs6_3x3x3.yaml"
    "training/ppo_masked/rs6_3x3x4.yaml"
    "training/ppo_masked/rs6_3x3x5.yaml"
    "training/ppo_masked/rs6_4x3x5.yaml"

    "training/ppo_masked/rs4_2x2x2.yaml"
    "training/ppo_masked/rs4_3x2x2.yaml"
    "training/ppo_masked/rs4_3x3x3.yaml"
    "training/ppo_masked/rs4_3x3x4.yaml"
    "training/ppo_masked/rs4_3x3x5.yaml"
    "training/ppo_masked/rs4_4x3x5.yaml"

#    "training/ppo_masked/rs8_2x2x2.yaml"
#    "training/ppo_masked/rs8_3x2x2.yaml"
#    "training/ppo_masked/rs8_3x3x3.yaml"
#    "training/ppo_masked/rs8_3x3x4.yaml"
#    "training/ppo_masked/rs8_3x3x5.yaml"
#    "training/ppo_masked/rs8_4x3x5.yaml"
)

for CONFIG_FILE in "${CONFIG_FILES[@]}"
do
    echo "Starting training with config $CONFIG_FILE at $(date)"
    python -m src.agents.train -fp "$CONFIG_FILE"
done

echo "All training runs completed at $(date)!"
