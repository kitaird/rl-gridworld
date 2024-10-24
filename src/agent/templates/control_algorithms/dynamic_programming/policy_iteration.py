from pathlib import Path

import yaml

from src.agent.common.policies import DeterministicPolicy
from src.agent.common.value_functions import StateValueFunction
from src.env.gym import Gym
from src.visualization.plotter import plot_value_function_sums


class PolicyIteration:
    """
        This is the dynamic programming policy iteration algorithm.
        Domain knowledge is available, meaning self.env can be used for planning.

        For reference, see Sutton & Barto, Reinforcement Learning: An Introduction, 2018, p. 80, chapter 4.3, Policy Iteration.
    """

    def __init__(self, env):
        self.algo_name: str = "Policy Iteration"
        with open(Path(__file__).parent.parent / 'algorithms-config.yml') as f:
            self.config = yaml.safe_load(f)[self.algo_name]
        self.env: Gym = env
        self.discount_factor: float = self.config['discount_factor']
        self._iterations: int = self.config['iterations']
        self._is_asynchronous: bool = self.config['is_asynchronous']
        self._plot_value_functions: bool = self.config['plot_value_functions']
        self._value_functions_sum: list[float] = []
        # This threshold makes sure that the policy evaluation ends after the improvements are too insignificant
        self._policy_evaluation_threshold: float = self.config['policy_evaluation_threshold']
        self.state_values: StateValueFunction = StateValueFunction(env=self.env, init_value=self.config['value_function_init'])
        self.policy: DeterministicPolicy = DeterministicPolicy(state_space=self.env.valid_states, action_space=self.env.actions)

    def clear(self) -> None:
        self.env.clear()
        self._value_functions_sum: list[float] = []
        self.state_values = StateValueFunction(env=self.env, init_value=self.config['value_function_init'])
        self.policy = DeterministicPolicy(state_space=self.env.valid_states, action_space=self.env.actions)

    def run(self) -> None:
        for _ in range(self._iterations):
            self.run_iteration()
        if self._plot_value_functions:
            self.render()

    def render(self) -> None:
        plot_value_function_sums(self.algo_name, self._value_functions_sum)

    def run_iteration(self) -> None:
        self.policy_evaluation_loop()
        self.policy_improvement()
        self._value_functions_sum.append(self.state_values.sum())

    def policy_evaluation_loop(self) -> None:
        evaluation_done = False
        while not evaluation_done:
            evaluation_done = self.evaluate_asynchronously() if self._is_asynchronous else self.evaluate_synchronously()

    def evaluate_asynchronously(self) -> bool:
        """
            TODO: Evaluate the current policy using bootstrapping.
            Compare each new state_value to the current state_value.
            Use self._policy_evaluation_threshold as a stopping condition when
            the value deltas become too small.

            This method performs an asynchronous update, meaning that the state_values are updated in place.
        """
        raise NotImplementedError()

    def evaluate_synchronously(self) -> bool:
        """
            TODO: Evaluate the current policy using bootstrapping.
            Compare each new state_value to the current state_value.
            Use self._policy_evaluation_threshold as a stopping condition when
            the value deltas become too small.

            This method performs a synchronous update, meaning that the state_values are updated all at once.
        """
        raise NotImplementedError()

    def policy_improvement(self) -> None:
        """
            TODO: Update self.policy for each state using the self.calculate_action_value() method.
        """
        raise NotImplementedError()

    def calculate_action_value(self, state, action) -> float:
        """
            TODO: Calculate the action_value of the given state-action-pair using planning.
            Remember the edge case when the next_state is terminal.
        """
        raise NotImplementedError()
