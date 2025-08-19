# Energy-Aware (Flexible) Job Shop Scheduling with Deep Reinforcement Learning

This repository extends the schlably framework (University of Wuppertal) with an Energy-Aware Multi-Objective FJSSP and a true direct action space where the agent selects both the job and the machine. It implements and evaluates the approach from the bachelor thesis “Solving an Energy-Aware Multi-Objective (Flexible) Job Shop Scheduling Problem using Deep Reinforcement Learning”.

For original schlably docs and tutorials, see the upstream documentation: https://schlably.readthedocs.io/en/latest/

## Overview
- Topic: Energy-aware multi-objective (flexible) job shop scheduling via deep RL.
- Goal: Minimize makespan and total energy consumption by learning scheduling policies.
- Base: Built on schlably; extended with energy modeling and a direct (job+machine) action space.

## Key Features
- Energy-aware FJSSP: Per-machine processing times and energy consumption for each task.
- Direct action space: Agent selects job and machine jointly (vs. job-only + heuristic machine).
- Multi-objective rewards: Reward strategies rs1–rs8 to trade off makespan and energy.
- Custom RL: Masked PPO with action masking and GAE; heuristic baselines for comparison.
- Visualization: Gantt charts, optional GIFs; benchmark parsing helpers.
- Reproducible: Config-driven; optional Weights & Biases logging; centralized energy utility.

## Your Contributions
- EnergyTask: Per-machine `processing_times` and `energy_consumptions`.
- EnergyEnv: Composite action space, energy-aware observations, and reward strategies `rs1..rs8`.
- Direct action space: Job+machine selection.
- Utilities: Shared `src/utils/energy.py` for canonical energy sampling; added a data-generation method (generate_energy_data/generate_energy_consumption) that creates energy values inversely related to processing time.
- Visualization: Energy-aware annotations in Gantt charts.
- Benchmarks: Integrated Fattahi et al. benchmark instances and extended them to include per-machine processing energy for each valid job–machine pair using the inverse-time energy model.
- Heuristics: Extended heuristic agent with FJSSP variants that select job–machine pairs (composite actions), not only jobs as in the standard schlably environment.
- End-to-end: Extended the complete stack—from data generation to testing—so energy-aware FJSSP models can be trained on energy-augmented data and evaluated against heuristics.

## Quickstart
- Python: 3.10 recommended.
- Install: `pip install -r requirements.txt`
- Train: `python -m src.agents.train -fp training/ppo_masked/energy_fjsssp_config_job3_task4_tools0.yaml`
- Test: `python -m src.agents.test -fp testing/ppo_masked/energy_fjsssp_config_job3_task4_tools0.yaml --plot-ganttchart`
- WandB: Optional; configure via config (`wandb_mode`) and `src/utils/logger.py`.

## Installation
- Dependencies: See `requirements.txt`.
- Setup: Create/activate a virtualenv; `pip install -r requirements.txt`.
- Optional: `wandb login` if using online logging.

## Data
- Prebuilt: Point `instances_file` in config to a dataset under `data/instances/...`.
- Generate: `python -m src.data_generator.instance_factory -fp data_generation/<your_config>.yaml`.
- Benchmarks: `src/data_generator/create_benchmark_instances.py` parses Fattahi et al. (SFJS) instances, augments them with per-machine energy using the inverse-time model, and visualizes them.

## Configs
- Training: `config/training/ppo_masked/...`
- Testing: `config/testing/ppo_masked/...`
- Data generation: `config/data_generation/...`
- Key fields: `environment` (e.g., `energy_env`), `algorithm` (e.g., `ppo_masked`), `reward_strategy` (`rs3` etc.), seeds, rollout sizes, `wandb_mode`, `instances_file`.

## Training
- Command: `python -m src.agents.train -fp config/training/ppo_masked/energy_fjssp_config_job3_task4_tools0.yaml`
- Logging: `src/utils/logger.py` (W&B optional). Intermediate tests via `intermediate_test_interval`.
- Models: Saved under `data/models/<saved_model_name>.pkl`.

## Testing
- Command: `python -m src.agents.test -fp config/testing/ppo_masked/energy_fjssp_config_job3_task4_tools0.yaml --plot-ganttchart`
- Baselines: Heuristics (EDD, SPT, MTR, LTR) run alongside the agent.
- Outputs: Metrics (reward, makespan, tardiness or energy) and optional Gantt images.

## Repository Structure
- `src/environments/`: Gym envs (`Env`, `IndirectActionEnv`, `EnergyEnv`) + `EnvironmentLoader`.
- `src/agents/`: Masked PPO, PPO, DQN; training/testing; heuristics; solver hooks.
- `src/data_generator/`: Generators for JSSP/FJSSP/Energy FJSSP; benchmark parsing; task classes.
- `src/visuals_generator/`: Gantt plotting and energy distribution visualization.
- `src/utils/`: Config/data/model handlers; logging; evaluation; energy utilities.

## Environments
- `Env` (Tetris-style FJSSP): Action is job index; machine chosen by earliest end-time; supports tools/tardiness.
- `IndirectActionEnv`: Indirect discrete action (0–9 → normalized runtime).
- `EnergyEnv` (Energy-aware FJSSP):
  - Action: `Discrete(num_jobs * num_machines)` decoded into `(job, machine)`.
  - Observation: Per-job features (min processing time, progress, per-machine energy) + global machine occupancy & progress.
  - Rewards: `rs1..rs8` to balance makespan and total energy (idle + processing).

## Action Spaces
- Job-only (legacy): Env selects machine via heuristic.
- Job+machine (direct): Agent controls both (in `EnergyEnv`).

## Reproducibility & Seeds
- Seeds: Set via config (`seed`); applied to NumPy, Python, and env (MaskedPPO).
- Deterministic sampling: `src/utils/energy.py` provides the canonical energy sampler used project-wide.

## Logging & Visualization
- W&B: `wandb_mode` (0=off, 1=offline, 2=online). Artifacts/tables supported.
- Gantt charts: `src/visuals_generator/gantt_chart.py` renders schedules; energy-aware annotations.

## Citations & Acknowledgements
- Thesis: “Solving an Energy-Aware Multi-Objective (Flexible) Job Shop Scheduling Problem using Deep Reinforcement Learning”.
- Base Framework: schlably (University of Wuppertal). Upstream docs at https://schlably.readthedocs.io/
- Base Benchmark Data: Fattahi et al. (SFJS) instances - doi:10.1007/s10845-007-0026-8
