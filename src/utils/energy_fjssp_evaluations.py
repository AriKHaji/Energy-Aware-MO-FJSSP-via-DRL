import numpy as np

from src.utils.evaluations import EvaluationHandler

class EvaluationHandlerForEnergyFJSSP(EvaluationHandler):
    """
    This subclass adapts the base EvaluationHandler to evaluate energy-related metrics.
    It records total energy consumption instead of tardiness, and computes energy-specific
    evaluation metrics.
    """
    def __init__(self):
        # Initialize the base evaluator attributes (e.g., rewards, makespan, tasks, actions)
        super().__init__()
        # Add a new list for energy consumption metrics
        self.total_energy_consumption = []
        # Optionally, if these metrics are not relevant here, you can set them to None or ignore them:
        self.tardiness = None
        self.tardiness_max = None

    def record_environment_episode(self, env, total_reward) -> None:
        """
        Overrides the base method to record energy consumption instead of tardiness.
        """
        self.makespan.append(env.get_makespan())
        self.rewards.append(total_reward)
        # Record the total energy consumption (assumed to be a list or iterable)
        self.total_energy_consumption.append(env.calculate_energy_consumption())
        self.tasks_list.append(env.tasks)
        self.actions_list.append(env.action_history)

    def update_episode_solved_with_solver(self, env) -> None:
        """
        Overrides the base method to compute missing parameters for an environment processed by a solver.
        This method now calls a function to calculate energy consumption.
        """
        env.calculate_energy_consumption()
        for task in env.tasks:
            if task.finished > env.ends_of_machine_occupancies[task.selected_machine]:
                env.ends_of_machine_occupancies[task.selected_machine] = task.finished
        self.record_environment_episode(env, 0)

    def evaluate_test(self) -> dict:
        """
        Computes evaluation metrics using the recorded rewards, makespan, and total energy consumption.
        """
        rewards = np.asarray(self.rewards)
        evaluation_results = {}
        evaluation_results['rew_mean'] = np.mean(rewards)
        evaluation_results['rew_std'] = np.std(rewards)
        evaluation_results['rew_best'] = np.max(rewards)
        evaluation_results['rew_best_count'] = sum([1 for el in rewards if el == evaluation_results['rew_best']])
        evaluation_results['rew_worst'] = np.min(rewards)
        evaluation_results['total_energy_consumption_mean'] = np.mean(self.total_energy_consumption)
        evaluation_results['total_energy_consumption_std'] = np.std(self.total_energy_consumption)
        evaluation_results['makespan_mean'] = np.mean(self.makespan)
        evaluation_results['rew_worst_quantile_border'] = np.quantile(rewards, 0.1)
        evaluation_results['rew_cvar'] = rewards[rewards <= evaluation_results['rew_worst_quantile_border']].mean()
        evaluation_results['rew_perc_good_solutions'] = 1 - np.count_nonzero(rewards) / len(rewards)
        evaluation_results['num_tests'] = len(rewards)
        return evaluation_results

    @classmethod
    def add_solver_gap_to_results(cls, results: dict) -> dict:
        """
        This method can remain unchanged if the way of computing the solver gap is the same.
        """
        if 'solver' in results:
            optimal_makespan = results['solver']['makespan_mean']
            for agent, result in results.items():
                gap = result['makespan_mean'] - optimal_makespan
                results[agent].update({'gap_to_solver': gap})
        return results
