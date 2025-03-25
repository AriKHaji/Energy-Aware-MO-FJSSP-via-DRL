"""
Helper function for the instance generation in instance_factory.py.
"""
# Standard package import
from enum import Enum
from typing import List, Tuple
import itertools
import random
from rich.console import Console
from rich.table import Table

# Library import
import numpy as np

from src.data_generator.energy_task import EnergyTask
# Functional internal import
from src.data_generator.task import Task


class SP(Enum):
    jssp = "_generate_instance_jssp"
    fjssp = "_generate_instance_fjssp"
    energy_fjssp = "_generate_instance_energy_fjssp"  # New sp_type


    @classmethod
    def is_sp_type_implemented(cls, sp_type: str = "") -> bool:
        return True if sp_type in cls.str_list_of_sp_types_implemented() else False

    @classmethod
    def str_list_of_sp_types_implemented(cls) -> List[str]:
        return [name for name, _ in cls.__members__.items()]


class SPFactory:

    @classmethod
    def generate_instances(cls, num_jobs: int = 2, num_tasks: int = 2, num_machines: int = 2, num_tools: int = 2,
                        num_instances: int = 2, runtimes: List[int] = None, sp_type: str = "jssp",
                        print_info: bool = False, **kwargs) -> List[List[Task]]:
        """
        Creates a list of instances with random values in the range of the input parameters

        :param num_jobs: number of jobs generated in an instance
        :param num_tasks: number of tasks per job generated in an instance
        :param num_machines: number of machines available
        :param num_tools: number of tools available
        :param num_instances: number of instances which are to be generated
        :param runtimes: list of possible runtimes for tasks
        :param sp_type: Scheduling problem type (e.g. "jssp")
        :param print_info: if True additional info printed to console

        :return: List of list of Task instances which together form an instance

        """

        # Default initialization of mutable parameters - code safety.
        if runtimes is None:
            runtimes = [4, 6]

        # Check if implemented SP type is provided and get matching generate instance function.
        assert SP.is_sp_type_implemented(sp_type), \
            f"{sp_type} is not valid, you have to provide a valid sp type: {SP.str_list_of_sp_types_implemented()}\n"
        generate_instance_function = getattr(cls, SP[sp_type].value)

        # Get possible combinations according to given parameters
        # Machines binary mask - permutations without all 0 element
        machines: List[Tuple] = list(itertools.product([0, 1], repeat=num_machines))[1:]
        # Tools diagonal matrix
        # TODO (minor) - fix typing - preset for else is bad
        tools: np.ndarray = np.eye(num_tools, dtype=int) if num_tools > 0 else [[]]
        # Generate all possible combinations for job and task
        comp_attributes_task: list = [machines, tools, runtimes]
        task_combinations = list(itertools.product(*comp_attributes_task))

        # Collect arguments for generate instance function
        current_kwargs = locals().copy()
        # Remove class argument from collection
        current_kwargs.pop('cls', None)
        # Remove additional kwargs from collection to pass them individually
        current_kwargs.pop('kwargs', None)

        instances = []
        # Create and collect n instances
        for _ in range(num_instances):
            # Call instance function with currently collected arguments
            new_instance = generate_instance_function(**current_kwargs, **kwargs)
            instances.append(new_instance)

        # Print information
        if print_info:
            print('Number of generated task combinations:', len(task_combinations))

        if sp_type == "energy_fjssp":
            visualize_instance(instances[0], num_jobs)
        else:
            visualize_regular_instance(instances[0], num_jobs)

        return instances

    @classmethod
    def _generate_instance_jssp(cls, task_combinations: List[Tuple[int]], num_jobs: int, num_tasks: int,
                             num_machines: int, num_tools: int, **kwargs) -> List[Task]:
        """
        Generates a jssp instance

        :param task_combinations: List with all possible tasks
        :param num_jobs: number of jobs generated in an instance
        :param num_tasks: number of tasks per job generated in an instance
        :param num_machines: number of machines available
        :param num_tools: number of tools available
        :param kwargs: Unused

        :return: jssp instance (List of tasks)

        """
        # Initial data_generator sanity check
        assert num_tasks == num_machines, "Warning: You are not creating a classical JSSP instance, " \
                                          "where num_machines = num_tasks must hold."

        instance = []
        # pick n jobs for this instance
        for j in range(num_jobs):
            # Generate random shuffled list of machines for job tasks
            machines_jssp_random = random.sample(list(np.arange(num_tasks)), num_tasks)
            # pick num_tasks tasks for this job
            for t in range(num_tasks):
                task = list(task_combinations[np.random.randint(0, len(task_combinations) - 1)])

                machines_jssp = [0 for _ in np.arange(num_tasks)]
                machines_jssp[machines_jssp_random[t]] = 1
                task[0] = tuple(machines_jssp)

                task = Task(
                    job_index=j,
                    task_index=t,
                    machines=list(task[0]),
                    tools=list(task[1]),
                    deadline=0,
                    done=False,
                    runtime=task[2],
                    _n_machines=num_machines,
                    _n_tools=num_tools
                )
                instance.append(task)
        return instance

    @classmethod
    def _generate_instance_fjssp(cls, task_combinations: List[Tuple[int]], num_jobs: int, num_tasks: int,
                              num_machines: int, num_tools: int, **kwargs) -> List[Task]:
        """
        Generates a fjssp instance

        :param task_combinations: List with all possible tasks
        :param num_jobs: number of jobs generated in an instance
        :param num_tasks: number of tasks per job generated in an instance
        :param num_machines: number of machines available
        :param num_tools: number of tools available
        :param kwargs: Unused

        :return: fjssp instance (List of tasks)

        """
        instance = []
        # pick n jobs for this instance
        for j in range(num_jobs):
            # pick num_tasks tasks for this job
            for t in range(num_tasks):
                task = list(task_combinations[np.random.randint(0, len(task_combinations) - 1)])
                task = Task(
                    job_index=j,
                    task_index=t,
                    machines=list(task[0]),
                    tools=list(task[1]),
                    deadline=0,
                    done=False,
                    runtime=task[2],
                    _n_machines=num_machines,
                    _n_tools=num_tools
                )
                instance.append(task)
        return instance

    @classmethod
    def _generate_instance_energy_fjssp(cls, task_combinations: List[Tuple[int]], num_jobs: int, num_tasks: int,
                                        num_machines: int, num_tools: int, **kwargs) -> List[EnergyTask]:
        """
        Generates an energy-aware fjssp instance with processing times sampled uniformly and energy
        consumption inversely related to processing times.
        """
        instance = []
        runtime_range = kwargs.get('runtime_range', (17, 214))  # global min/max runtime values
        e_min = kwargs.get('e_min', 20)
        e_max = kwargs.get('e_max', 220)
        alpha = kwargs.get('alpha', 4)
        random_seed = kwargs.get('random_seed', None)

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        for j in range(num_jobs):
            for t in range(num_tasks):
                # Pick a random task combination for tool requirements.
                task_tuple = list(task_combinations[np.random.randint(0, len(task_combinations) - 1)])
                tools_value = list(task_tuple[1])

                # Randomly determine the number of available machines (between 1 and num_machines).
                x = random.randint(1, num_machines)
                available_machines = sorted(random.sample(range(num_machines), x))

                # Create a binary mask of length num_machines.
                binary_mask = [1 if i in available_machines else 0 for i in range(num_machines)]

                # Sample processing times uniformly within runtime_range.
                processing_times_list = np.random.randint(runtime_range[0], runtime_range[1] + 1, size=x)
                processing_times_mapping = dict(zip(available_machines, processing_times_list))

                # Generate energy consumption values based on processing times.
                energy_values = generate_energy_consumption(
                    processing_times_list,
                    e_min=e_min,
                    e_max=e_max,
                    alpha=alpha,
                    random_seed=random_seed
                )
                energy_mapping = dict(zip(available_machines, energy_values))

                # Create the Task instance with the binary mask.
                task = EnergyTask(
                    job_index=j,
                    task_index=t,
                    machines=binary_mask,
                    tools=tools_value,
                    deadline=0,
                    done=False,
                    _n_machines=num_machines,
                    _n_tools=num_tools,
                    energy_consumptions=energy_mapping,
                    processing_times=processing_times_mapping
                )

                instance.append(task)

        return instance

    @classmethod
    def set_deadlines_to_max_deadline_per_job(cls, instances: List[List[Task]], num_jobs: int):
        """
        Equals all Task deadlines from one Job according to the one of the last task in the job

        :param instances: List of instances
        :param num_jobs: Number of jobs in an instance

        :return: List of instances with equaled job deadlines

        """
        # Argument sanity check
        assert isinstance(instances, list) and isinstance(num_jobs, int), \
            "Warning: You can only set deadlines for a list of instances with num_jobs of type integer."

        for instance in instances:
            # reset max deadlines for current instance
            max_deadline = [0] * num_jobs
            # get max deadline for every job
            for task in instance:
                if task.deadline > max_deadline[task.job_index]:
                    max_deadline[task.job_index] = task.deadline
            # assign max deadline to every task
            for task in instance:
                task.deadline = max_deadline[task.job_index]

    @classmethod
    def compute_and_set_hashes(cls, instances: List[List[Task]]):
        for instance in instances:
            instance_hash = hash(tuple(instance))
            # set hash attributes of each task
            for task in instance:
                task.instance_hash = instance_hash

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


def visualize_regular_instance(instance: list, num_jobs: int):
    """
    Visualizes a single instance of regular tasks (without energy consumption data) using a rich table.

    instance: List of Task objects.
    num_jobs: Total number of jobs in this instance.
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
        table.add_column("Runtime", justify="center")

        for task in sorted(tasks, key=lambda t: t.task_index):
            machine_mask = str(task.machines)
            runtime = str(task.runtime)
            table.add_row(str(task.task_index), machine_mask, runtime)

        console.print(table)


def generate_energy_consumption(processing_times, e_min, e_max, alpha=2, random_seed=None):
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


if __name__ == '__main__':
    # my_instances = SPFactory.generate_instances(num_jobs=4, num_tasks=5, num_machines=4, sp_type="fjssp")
    # for i, my_instance in enumerate(my_instances):
    #     print("Setup", i)
    #     for my_task in my_instance:
    #         print(my_task.str_info())
    pass
