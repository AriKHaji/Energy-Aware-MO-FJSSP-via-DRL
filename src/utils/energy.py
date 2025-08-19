"""
Energy-related utility functions shared across data generation and visualization.
"""
from __future__ import annotations

import numpy as np


def generate_energy_consumption(
    p: int,
    e_min: int,
    e_max: int,
    p_min: int,
    p_max: int,
    alpha: int = 4,
    random_seed: int | None = None,
) -> int:
    """
    Sample an integer energy consumption for a single processing time p using a
    Beta-based distribution inversely related to processing time.

    Parameters
    - p: processing time
    - e_min/e_max: energy range
    - p_min/p_max: global processing time range used for normalization
    - alpha: shape parameter controlling randomness
    - random_seed: if provided, seeds numpy once for deterministic sampling

    Returns
    - integer energy consumption
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Normalize processing time into [0, 1]
    p_norm = (p - p_min) / (p_max - p_min)

    # Beta parameters: higher p => higher b (lower mean), lower p => higher a (higher mean)
    a_param = 1 + alpha * (1 - p_norm)
    b_param = 1 + alpha * p_norm

    # Sample and scale to energy range
    energy = e_min + (e_max - e_min) * np.random.beta(a_param, b_param)
    return int(np.round(energy))

