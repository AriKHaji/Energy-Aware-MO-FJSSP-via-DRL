import wandb
from typing import List
from src.utils.logger import Logger

class LoggerForEnergyFJSSP(Logger):
    """
    This class can store various parameters (e.g. loss from a model update) as key value pairs.
    By calling the dump function, all stored parameters are logged according to the log_mode.
    Because the logger supports wandb there are several functions to initialize and handle the logging to wandb

    :param: config: Training config

    """
    def __init__(self, config: dict):
        super().__init__(config)

        self.WANDB_TABLE_COLUMNS: List[str] = ["Agent", "Reward", "Makespan", "Total_Energy_Consumption", "Ganttchart"]
        self.WANDB_FINAL_EVALUATION_TABLE_COLUMNS: List[str] = ['Agent', 'Mean Reward', 'Mean Makespan', 'Mean Total_Energy_Consumption']

    def write_to_wandb_summary(self, evaluation_results: dict):
        """
        Log results as summary to wandb

        :param evaluation_results: Dictionary with at least all evaluation result to be logged in this function
        :return: None

        """
        if self.wandb_run:
            final_evaluation_table = wandb.Table(columns=self.WANDB_FINAL_EVALUATION_TABLE_COLUMNS)
            # iterate overall all agent whose results are saved in evaluation_results
            for agent in evaluation_results.keys():
                log_data = []
                log_data.append(str(agent))
                log_data.append(evaluation_results[agent]['rew_mean'])
                log_data.append(evaluation_results[agent]['makespan_mean'])
                log_data.append(evaluation_results[agent]['total_energy_consumption_mean'])
                final_evaluation_table.add_data(*log_data)
            self.wandb_run.log({'Final Evaluation Table': final_evaluation_table})