from typing import List
from src.data_generator.task import Task  # Import the original Task class


class EnergyTask(Task):
    """
    A Task subclass that includes an energy consumption mapping.

    Each task can be processed on a subset of machines, and this mapping stores the energy
    consumption per time unit for each available machine.
    """

    def __init__(self,
                 job_index: int,
                 task_index: int,
                 machines: List[int] = None,
                 tools: List[int] = None,
                 deadline: int = None,
                 instance_hash: int = None,
                 done: bool = None,
                 runtime: int = None,
                 started: int = None,
                 finished: int = None,
                 selected_machine: int = None,
                 _n_machines: int = None,
                 _n_tools: int = None,
                 _feasible_machine_from_instance_init: int = None,
                 _feasible_order_index_from_instance_init: int = None,
                 energy_consumptions: dict = None,
                 processing_times: dict = None
                 ):
        # Initialize the base Task class attributes
        super().__init__(job_index=job_index,
                         task_index=task_index,
                         machines=machines,
                         tools=tools,
                         deadline=deadline,
                         instance_hash=instance_hash,
                         done=done,
                         runtime=runtime,
                         started=started,
                         finished=finished,
                         selected_machine=selected_machine,
                         _n_machines=_n_machines,
                         _n_tools=_n_tools,
                         _feasible_machine_from_instance_init=_feasible_machine_from_instance_init,
                         _feasible_order_index_from_instance_init=_feasible_order_index_from_instance_init)

        # Store the energy consumption of the operation on each available machine
        # (e.g., {machine_index: energy_value})
        self.energy_consumptions = energy_consumptions
        # Store the processing time of the operation for each available machine
        # (e.g., {machine_index: processing_time})
        self.processing_times = processing_times

    def __str__(self) -> str:
        base_str = super().__str__()
        return base_str + f" | With available machines: {self.machines}, Energy consumptions: {self.energy_consumptions} and Processing times: {self.processing_times}"

