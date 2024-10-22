from pathlib import Path

import numpy as np
import yaml

from src.agent.common.policies import create_epsilon_soft_policy, Policy
from src.agent.common.value_functions import ActionValueFunction
from src.env.action import Action
from src.env.gym import Gym
from src.env.state import State
from src.visualization.plotter import plot_value_function_sums, plot_returns


class NStepSarsa:
    """
        Implements the n-step Sarsa algorithm.

        This algorithm updates the action value function based on n-step returns.
        It follows an epsilon-soft policy and uses a decaying epsilon for exploration.

        For reference, see Sutton & Barto, Reinforcement Learning: An Introduction, 2018, p. 145, chapter 7.2, n-step Sarsa.
    """

    def __init__(self, env):
        self.algo_name: str = "n-step Sarsa"
        with open(Path(__file__).parent.parent / 'algorithms-config.yml') as f:
            self.config = yaml.safe_load(f)[self.algo_name]
        self.env: Gym = env
        self.discount_factor: float = self.config['discount_factor']
        self.step_size: float = self.config['step_size']
        self.epsilon: float = self.config['epsilon']
        self.epsilon_decay: float = self.config['epsilon_decay']
        self._total_episodes: int = 0
        self.n: int = self.config['n']
        self._iterations: int = self.config['iterations']
        self._rng = np.random.default_rng(seed=1)
        self._plot_returns: bool = self.config['plot_returns']
        self._plot_value_functions: bool = self.config['plot_value_functions']
        self._returns: list[float] = []
        self._value_functions_sum: list[float] = []
        self.action_values: ActionValueFunction = ActionValueFunction(env=self.env, init_value=self.config['value_function_init'])
        self.policy: Policy = create_epsilon_soft_policy(action_value_function=self.action_values, epsilon=self.epsilon)

    def clear(self) -> None:
        self.env.clear()
        self._returns = []
        self._total_episodes = 0
        self._value_functions_sum = []
        self.epsilon = self.config['epsilon']
        self.action_values = ActionValueFunction(env=self.env, init_value=self.config['value_function_init'])
        self.policy = create_epsilon_soft_policy(action_value_function=self.action_values, epsilon=self.epsilon)

    def run(self) -> None:
        for _ in range(self._iterations):
            self.run_iteration()
        if self._plot_returns:
            self.render()

    def render(self) -> None:
        if self._plot_value_functions:
            plot_value_function_sums(self.algo_name, self._value_functions_sum)
        if self._plot_returns:
            plot_returns(self.algo_name, self._returns)

    def run_iteration(self) -> None:
        self.run_control_loop()
        self._value_functions_sum.append(self.action_values.sum())

    def run_control_loop(self) -> None:
        """
            TODO: Run the control loop for the n-step Sarsa algorithm.
            At the end, append the total return of the episode to the returns list for plotting.
        """
        raise NotImplementedError()

    def calculate_return(self, T, stored_actions, stored_rewards, stored_states, tau) -> float:
        """
            TODO: Calculate the return G_tau:tau+n.
        """
        raise NotImplementedError()

    def update_action_value(self, state: State, action: Action, G: float) -> None:
        """
            TODO: Update the action value function using the TD-Learning update rule.
        """
        raise NotImplementedError()

    def calculate_action_value(self, s: State, a: Action, G: float) -> float:
        """
            TODO: Calculate the action value.
        """
        raise NotImplementedError()
