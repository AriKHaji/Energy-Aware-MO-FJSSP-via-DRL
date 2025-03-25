import sys

import gym
import numpy
from gym import spaces
import numpy as np
import copy

from torch.distributed.rpc import new_method

from src.data_generator.energy_task import EnergyTask
from src.visuals_generator.gantt_chart import GanttChartPlotter
from typing import List, Tuple, Dict, Any, Union

class EnergyEnv(gym.Env):
    """
    Environment for scheduling optimization.
    This class inherits from the base gym environment, so the functions step, reset, _state_obs and render
    are implemented and can be used by default.

    This environment is tailored for FJSSP with the optimization objectives of makespan and total energy consumption.

    """

    def __init__(self, config: dict, data: list):
        # Call the base class constructor
        super(EnergyEnv, self).__init__()
        # You can also add new parameters from the config if needed,
        # for example, an energy weight coefficient.

        self.energy_weight = config.get('energy_weight', 0.5)



        # import data containing all instances
        self.data: List[List[EnergyTask]] = data  # is later shuffled before input into the environment

        # get number of jobs, tasks, tools, machines and runtimes from input data
        self.num_jobs, self.num_tasks, self.max_runtime, self.max_energy = self.get_instance_info()
        self.num_machines: int = copy.copy(self.data[0][0]._n_machines)
        self.num_tools: int = copy.copy(self.data[0][0]._n_tools)
        self.num_all_tasks: int = self.num_jobs * self.num_tasks
        self.num_steps_max: int = config.get('num_steps_max', self.num_all_tasks)
        self.max_task_index: int = self.num_tasks - 1
        self.max_job_index: int = self.num_jobs - 1

        self.idle_range = config.get('machine_idle_energy_range', [0, 0.5, 1, 1.5])

        # retrieve run-dependent settings from config
        self.shuffle: bool = config.get('shuffle', False)
        self.log_interval: int = config.get('log_interval', 10)

        # initialize info which is reset by the reset-method after every episode
        self.num_steps: int = 0
        self.makespan: int = 0
        self.total_energy_consumption: float = 0
        self.ends_of_machine_occupancies: numpy.ndarray = np.zeros(self.num_machines, dtype=int)
        self.job_task_state: numpy.ndarray = np.zeros(self.num_jobs, dtype=int)
        self.task_job_mapping: dict = {}
        self.machine_idle_energys = np.random.uniform(self.idle_range[0], self.idle_range[1], size=self.num_machines)

        # initialize info which is not reset
        self.runs: int = -2  # counts runs (episodes/dones).  -1 because reset is called twice before start
        self.last_mask: numpy.ndarray = np.zeros(self.num_jobs)
        self.tasks: List[EnergyTask] = []
        self.data_idx: int = 0
        self.iterations_over_data = -1

        # training info log updated after each "epoch" over all training data
        self.action_history: List = []  # stores the sequence of tasks taken
        self.executed_job_history: List = []  # stores the sequence of jobs, of which the task is scheduled
        self.reward_history: List = []  # stores the rewards
        self.episodes_rewards: List = []
        self.episodes_makespans: List = []
        self.episodes_total_energy_consumptions: List = []

        # logging info buffers. Are reset in self.log_intermediate_step
        self.logging_makespans: List = []
        self.logging_rewards: List = []
        self.logging_total_energy_consumptions = []


        # action_space: idx_job
        self.action_space: spaces.Discrete = spaces.Discrete(self.num_jobs * self.num_machines)


        # initial observation
        self._state_obs: List = self.reset()

        # observation space
        observation_shape = np.array(self.state_obs).shape
        self.observation_space: spaces.Box = spaces.Box(low=0, high=1, shape=observation_shape)

        # reward parameters
        self.reward_strategy = config.get('reward_strategy')
        self.reward_scale = config.get('reward_scale', 1)
        self.mr2_reward_buffer: List[List] = [[] for _ in range(len(data))]  # needed for m2r reward only

    def reset(self) -> List[float]:
        """
        - Resets the episode information trackers
        - Updates the number of runs
        - Loads new instance

        :return: First observation by calling the class function self.state_obs

        """
        # update runs (episodes passed so far)
        self.runs += 1

        # reset episode counters and infos
        self.num_steps = 0
        self.total_energy_consumption = 0
        self.makespan = 0
        self.ends_of_machine_occupancies = np.zeros(self.num_machines, dtype=int)
        self.job_task_state = np.zeros(self.num_jobs, dtype=int)
        self.action_history = []
        self.executed_job_history = []
        self.reward_history = []
        self.machine_idle_energys = np.random.uniform(self.idle_range[0], self.idle_range[1], size=self.num_machines)

        # clear episode rewards after all training data has passed once. Stores info across runs.
        if self.data_idx == 0:
            self.episodes_makespans, self.episodes_rewards, self.episodes_total_energy_consumptions = ([], [], [])

        # load new instance every run
        self.data_idx = self.runs % len(self.data)
        self.tasks = copy.deepcopy(self.data[self.data_idx])
        if self.shuffle:
            np.random.shuffle(self.tasks)
        self.task_job_mapping = {(task.job_index, task.task_index): i for i, task in enumerate(self.tasks)}


        return self.state_obs

    def step(self, action: Union[int, float], **kwargs) -> (List[float], Any, bool, Dict):
        """
        Step Function
        :param action: Action to be performed on the current state of the environment
        :return: Observation, reward, done, infos
        """
        # transform and store action
        job_id, machine_id = self.decode_action(action)
        self.action_history.append(action)

        # check if the action is valid/executable
        if self.check_valid_action(job_id, machine_id):
            # if the action is valid/executable/schedulable
            selected_task_id, selected_task = self.get_selected_task(job_id)
            selected_machine = machine_id
            self.execute_action(job_id, selected_task, selected_machine)


            # print("KORREKT: Action {} is valid".format(action))
            # print("    Ausgewählter Task:" + selected_task.__str__())
            # print("    Task ist fertig oder nicht")
            # print("    Ausgewählte Machine:" + machine_id.__str__())

        else:
            # if the action is not valid/executable/schedulable

            selected_task_id, selected_task = self.get_selected_task(job_id)
            # print("FEHLER: Action {} is not valid".format(action))
            # print("    Ausgewählter Job:" + selected_task.__str__())
            # print("    Ausgewählter Machine:" + machine_id.__str__())

            #sys.exit()
            pass

        # update variables and track reward
        action_mask = self.get_action_mask()
        infos = {'mask': action_mask}
        observation = self.state_obs
        reward = self.compute_reward()
        self.reward_history.append(reward)

        done = self.check_done()
        if done:
            episode_reward_sum = np.sum(self.reward_history)
            makespan = self.get_makespan()
            total_energy_consumption = self.calculate_energy_consumption()

            self.episodes_makespans.append(self.get_makespan())
            self.episodes_rewards.append(np.mean(self.reward_history))
            self.episodes_total_energy_consumptions.append(total_energy_consumption)

            self.logging_rewards.append(episode_reward_sum)
            self.logging_makespans.append(makespan)
            self.logging_total_energy_consumptions.append(total_energy_consumption)

            if self.runs % self.log_interval == 0:
                self.log_intermediate_step()

        self.num_steps += 1
        return observation, reward, done, infos

    @property
    def state_obs(self) -> List[float]:
        obs = []

        # Ensure a maximum runtime is defined for normalization.
        if not hasattr(self, 'max_runtime'):
            self.max_runtime = max(task.runtime for task in self.data[0]) if self.data[0] else 1

        # --- Local Features per Job ---
        for job in range(self.num_jobs):
            # Determine the index of the next task for the job.
            t_idx = self.job_task_state[job] if self.job_task_state[job] < self.max_task_index else self.max_task_index
            next_task = copy.copy(self.tasks[self.task_job_mapping[(job, t_idx)]])

            # Normalized runtime (using the minimum processing time among allowed machines)
            runtime_norm = min(next_task.processing_times.values()) / self.max_runtime

            # Normalized task index (progress)
            task_index_norm = next_task.task_index / (self.num_tasks - 1) if self.num_tasks > 1 else 0

            # Machine-specific processing energy vector.
            machine_energy = []
            for m in range(self.num_machines):
                if next_task.machines[m] == 1:
                    proc_time = next_task.processing_times.get(m, 0)
                    energy_cons = next_task.energy_consumptions.get(m, 0)
                    energy_val = proc_time * energy_cons
                    machine_energy.append(energy_val)
                else:
                    machine_energy.append(0.0)

            # Normalize energy vector per job.
            machine_energy_norm = [e / self.max_energy for e in machine_energy]

            # Append local features for the job.
            obs.extend([runtime_norm, task_index_norm] + machine_energy_norm)

        # --- Global Features ---

        # 1. Machine Occupancy / Availability:
        current_makespan = self.get_makespan()
        # Normalize each machine's last occupancy by the current makespan.
        norm_availability = [
            occ / current_makespan if current_makespan > 0 else 0
            for occ in self.ends_of_machine_occupancies
        ]
        obs.extend(norm_availability)

        # 2. Idle Energy Costs:
        # Compute idle time for each machine (time gap from last task finish to current makespan).
        idle_times = [
            max(0, current_makespan - self.ends_of_machine_occupancies[m])
            for m in range(self.num_machines)
        ]
        # Multiply idle time by the machine's idle energy consumption to get cost.
        idle_energy_costs = [
            idle * rate for idle, rate in zip(idle_times, self.machine_idle_energys)
        ]
        # Normalize idle energy costs.
        max_idle_energy = max(idle_energy_costs) if max(idle_energy_costs) > 0 else 1.0
        norm_idle_energy_costs = [cost / max_idle_energy for cost in idle_energy_costs]
        obs.extend(norm_idle_energy_costs)

        # 3. Global Progress Metrics:
        # Normalized makespan (using max_runtime as a rough normalizer; adjust as needed).
        norm_makespan = current_makespan / self.max_runtime
        # Normalized cumulative energy consumption.
        # (Here you might need to choose a normalization factor based on the expected range.)
        norm_total_energy = self.total_energy_consumption / (
                    current_makespan * np.mean(self.machine_idle_energys) + 1e-6)
        obs.extend([norm_makespan, norm_total_energy])

        self._state_obs = obs
        return self._state_obs

    def decode_action(self, action: int) -> Tuple[int, int]:
        """
        Decodes the composite action into a job index and a machine index.

        Given that the action space is defined as num_jobs * num_machines,
        we decode the action using integer division and modulo operations:

            job_index = action // self.num_machines
            machine_index = action % self.num_machines

        :param action: Composite action integer (0 to num_jobs*num_machines - 1)
        :return: Tuple of (job_index, machine_index)
        """
        job_index = action // self.num_machines
        machine_index = action % self.num_machines
        return job_index, machine_index

    def get_action_mask(self) -> np.array:
        """
        Returns a composite action mask of shape (num_jobs * num_machines).
        For each (job, machine) pair:
          - 1 if the job has pending tasks and the next task is allowed on that machine.
          - 0 otherwise.
        """
        mask = []
        for job in range(self.num_jobs):
            # Check if job has pending tasks.
            job_available = self.job_task_state[job] < self.num_tasks
            for machine in range(self.num_machines):
                if not job_available:
                    mask.append(0)
                else:
                    # Get the next task for the job.
                    _, task = self.get_selected_task(job)
                    # Check if the machine is allowed for this task.
                    mask.append(1 if task.machines[machine] == 1 else 0)
        mask = np.array(mask)
        self.last_mask = mask
        return mask

    def check_valid_action(self, job_index: int, machine_index: int) -> bool:
        job_mask = self.get_action_mask()  # composite mask of length num_jobs * num_machines
        if job_mask[job_index * self.num_machines + machine_index] == 0:
            return False

        _, task = self.get_selected_task(job_index)
        return task.machines[machine_index] == 1

    def execute_action(self, job_id: int, task: EnergyTask, machine_id: int) -> None:
        """
        This Function executes a valid action
        - set machine
        - update job and task

        :param job_id: job_id of the task to be executed
        :param task: Task
        :param machine_id: ID of the machine on which the task is to be executed

        :return: None

        """

        """
        print("This function executes a valid action")
        print("    a valid action is: " + task.__str__() + ", machine id: " + machine_id.__str__())
        print("    The " + task.__str__() + " could also run on the machines " + task.machines.__str__() )
        invalid_machine_actions = 0
        if not task.machines[machine_id]:
            invalid_machine_actions += 1
        print("Invalid Machine Count: " + str(invalid_machine_actions))
        """

        task.runtime = task.processing_times.get(machine_id)

        # check task preceding in the job (if it is not the first task within the job)
        if task.task_index == 0:
            start_time_of_preceding_task = 0
        else:
            preceding_task = self.tasks[self.task_job_mapping[(job_id, task.task_index - 1)]]
            start_time_of_preceding_task = preceding_task.finished

        # check earliest possible time to schedule according to preceding task and needed machine
        start_time = max(start_time_of_preceding_task, self.ends_of_machine_occupancies[machine_id])

        end_time = start_time + task.runtime

        # update machine occupancy and job_task_state
        self.ends_of_machine_occupancies[machine_id] = end_time
        self.job_task_state[job_id] += 1

        # update job and task
        task.started = start_time
        task.finished = end_time
        task.selected_machine = machine_id
        task.done = True

    def get_selected_task(self, job_idx: int) -> Tuple[int, EnergyTask]:
        """
        Helper Function to get the selected task (next possible task) only by the job index

        :param job_idx: job index

        :return: Index of the task in the task list and the selected task

        """
        task_idx = self.task_job_mapping[(job_idx, self.job_task_state[job_idx])]
        selected_task = self.tasks[task_idx]
        return task_idx, selected_task

    def log_intermediate_step(self) -> None:
        """
        Log Function

        :return: None

        """
        if self.runs >= self.log_interval:
            print('-' * 110, f'\n{self.runs} instances played! Last instance seen: {self.data_idx}/{len(self.data)}')
            print(f'Average performance since last log: mean reward={np.around(np.mean(self.logging_rewards), 2)}, '
                     f'mean makespan={np.around(np.mean(self.logging_makespans), 2)}, '
                     f'mean energy consumption={np.around(np.mean(self.logging_total_energy_consumptions), 2)}')
            self.logging_rewards.clear()
            self.logging_makespans.clear()
            self.logging_total_energy_consumptions.clear()

    def get_makespan(self):
        """
        Returns the current makespan (the time the latest of all scheduled tasks finishes)
        """
        return np.max(self.ends_of_machine_occupancies)

    def calculate_machine_idle_time(self, machine_id: int) -> float:
        """
        Calculates the total idle time for a given machine in the current schedule.

        :param machine_id: The index of the machine.
        :return: Total idle time for that machine.
        """
        # Gather tasks that were executed on the specified machine.
        machine_tasks = [task for task in self.tasks if task.selected_machine == machine_id]

        # If no task is scheduled on this machine, it is idle for the entire duration.
        if not machine_tasks:
            return self.get_makespan()

        # Sort the tasks by their start time.
        machine_tasks.sort(key=lambda task: task.started)

        idle_time = 0.0
        # Idle time before the first task.
        idle_time += machine_tasks[0].started

        # Idle time between tasks.
        for i in range(1, len(machine_tasks)):
            idle_interval = machine_tasks[i].started - machine_tasks[i - 1].finished
            idle_time += max(0, idle_interval)

        # Idle time after the last task until the makespan.
        makespan = self.get_makespan()
        idle_time += max(0, makespan - machine_tasks[-1].finished)

        return idle_time

    def calculate_processing_energy(self) -> float:
        total_processing_energy = 0.0

        scheduled_tasks = [task for task in self.tasks if task.selected_machine is not None]
        for task in scheduled_tasks:
            total_processing_energy += task.energy_consumptions.get(task.selected_machine)

        return total_processing_energy

    def calculate_energy_consumption(self) -> float:

        total_idle_energy = 0.0
        for i in range(self.num_machines):
            total_idle_energy += self.calculate_machine_idle_time(i) * self.machine_idle_energys[i]

        total_processing_energy = self.calculate_processing_energy()

        return total_idle_energy + total_processing_energy


    def check_done(self) -> bool:
        """
        Check if all jobs are done

        :return: True if all jobs are done, else False

        """
        sum_done = sum([task.done for task in self.tasks])
        return sum_done == self.num_all_tasks or self.num_steps == self.num_steps_max

    def get_instance_info(self) -> (int, int, int, int):
        """
        Retrieves info about the instance size and configuration from an instance sample
        :return: (number of jobs, number of tasks and the maximum runtime) of this datapoint
        """
        num_jobs, num_tasks, max_runtime, max_energy = 0, 0, 0, 0
        for task in self.data[0]:
            num_jobs = task.job_index if task.job_index > num_jobs else num_jobs
            num_tasks = task.task_index if task.task_index > num_tasks else num_tasks
            max_runtime = max(task.processing_times.values()) if max(task.processing_times.values()) > max_runtime else max_runtime
            max_energy = max(task.energy_consumptions.values()) if max(task.energy_consumptions.values()) > max_energy else max_runtime
        return num_jobs + 1, num_tasks + 1, max_runtime, max_energy

    def close(self):
        """
        This is a relict of using OpenAI Gym API. This is currently unnecessary.
        """
        pass

    def seed(self, seed=1):
        """
        This is a relict of using OpenAI Gym API.
        Currently unnecessary, because the environment is deterministic -> no seed is used.
        """
        return seed

    def render(self, mode='human'):
        """
        Visualizes the current status of the environment

        :param mode: "human": Displays the gantt chart,
                     "image": Returns an image of the gantt chart

        :return: PIL.Image.Image if mode=image, else None

        """
        if mode == 'human':
            GanttChartPlotter.get_gantt_chart_image(self.tasks, show_image=True, return_image=False)
        elif mode == 'image':
            return GanttChartPlotter.get_gantt_chart_image(self.tasks)
        else:
            raise NotImplementedError(f"The Environment on which you called render doesn't support mode: {mode}")


    def compute_reward(self) -> float:
        """
        Calculates the reward that will later be returned to the agent. Uses the self.reward_strategy string to
        discriminate between different reward strategies. Default is 'rs3'.

        :return: Reward

        """
        prev_makespan = self.makespan
        new_makespan = self.get_makespan()
        self.makespan = new_makespan

        prev_total_energy_consumption = self.total_energy_consumption
        new_total_energy_consumption = self.calculate_energy_consumption()
        s = 1 / (self.num_jobs * self.num_machines) # Scaling Factor for Energy Consumption
        self.total_energy_consumption = new_total_energy_consumption

        reward = 0

        if self.reward_strategy == 'rs1':
            reward = self.compute_rs1(prev_makespan, new_makespan, prev_total_energy_consumption, new_total_energy_consumption, s)
        elif self.reward_strategy == 'rs2':
            reward = self.compute_rs2(prev_makespan, new_makespan, prev_total_energy_consumption, new_total_energy_consumption, s)
        elif self.reward_strategy == 'rs3':
            reward = self.compute_rs3(prev_makespan, new_makespan, prev_total_energy_consumption, new_total_energy_consumption, s)
        elif self.reward_strategy == 'rs4':
            reward = self.compute_rs4(prev_makespan, new_makespan, prev_total_energy_consumption, new_total_energy_consumption, s)
        elif self.reward_strategy == 'rs5':
            reward = self.compute_rs5(prev_makespan, new_makespan, prev_total_energy_consumption, new_total_energy_consumption, s)
        elif self.reward_strategy == 'rs6':
            reward = self.compute_rs6(prev_makespan, new_makespan, prev_total_energy_consumption, new_total_energy_consumption, s)

        reward *= self.reward_scale

        return reward


    def compute_rs1(self, prev_makespan, new_makespan, prev_total_energy_consumption, new_total_energy_consumption, s) -> float:
        return prev_makespan - new_makespan

    def compute_rs2(self, prev_makespan, new_makespan, prev_total_energy_consumption, new_total_energy_consumption, s) -> float:
        reward1 = prev_makespan - new_makespan
        reward2 = prev_total_energy_consumption - new_total_energy_consumption
        return reward1 + s * reward2

    def compute_rs3(self, prev_makespan, new_makespan, prev_total_energy_consumption, new_total_energy_consumption, s) -> float:
        reward1 = - ((prev_makespan - new_makespan) ** 2)
        reward2 = - ((prev_total_energy_consumption - new_total_energy_consumption) ** 2)
        return reward1 + s * reward2

    def compute_rs4(self, prev_makespan, new_makespan, prev_total_energy_consumption, new_total_energy_consumption, s) -> float:
        reward1 = -(prev_makespan - new_makespan) ** 2
        reward2 = prev_total_energy_consumption - new_total_energy_consumption

        # Additionally, Reward for parallelization by equal machine workload
        workloads = self.ends_of_machine_occupancies  # an array of finish times per machine
        std_workload = np.std(workloads)
        additional_reward = - (std_workload ** 2)  # Squared so that larger differences are penalized more

        return reward1 + s * reward2 + additional_reward

    def compute_rs5(self, prev_makespan, new_makespan, prev_total_energy_consumption, new_total_energy_consumption, s) -> float:
        reward1 = -((prev_makespan - new_makespan) ** 2)
        reward2 = -((prev_total_energy_consumption - new_total_energy_consumption) ** 2)

        reward = reward1 + s * reward2

        if self.check_done():
            reward += - new_makespan - s * new_total_energy_consumption

        return reward

    def compute_rs6(self, prev_makespan, new_makespan, prev_total_energy_consumption, new_total_energy_consumption, s) -> float:
        if not self.check_done():
            return 0
        else:
            reward1 = - new_makespan
            reward2 = - new_total_energy_consumption
            return reward1 + s * reward2
