from pathlib import Path

import numpy as np
import yaml

from src.agent.common.env_model import DeterministicEnvModel
from src.agent.common.policies import Policy, create_epsilon_soft_policy
from src.agent.common.value_functions import ActionValueFunction
from src.env.action import Action
from src.env.gym import Gym
from src.env.state import State
from src.visualization.plotter import plot_returns, plot_value_function_sums


class DynaQ:
    """
        Dyna-Q algorithm implementation.

        This class implements the Dyna-Q algorithm, which combines model-free and model-based reinforcement learning.
        The agent learns from real experiences and uses a learned model of the environment to simulate additional experiences.

        For reference, see Sutton & Barto, Reinforcement Learning: An Introduction, 2018, p. 161, chapter 8.2, Dyna: Integrated Planning, Acting and Learning.
    """

    def __init__(self, env):
        self.algo_name: str = "Dyna-Q"
        with open(Path(__file__).parent.parent / 'algorithms-config.yml') as f:
            self.config = yaml.safe_load(f)[self.algo_name]
        self.env: Gym = env
        self.discount_factor: float = self.config['discount_factor']
        self.step_size: float = self.config['step_size']
        self.planning_steps: int = self.config['planning_steps']
        self.epsilon: float = self.config['epsilon']
        self.epsilon_decay: float = self.config['epsilon_decay']
        self._total_episodes: int = 0
        self._iterations: int = self.config['iterations']
        self._rng = np.random.default_rng(seed=1)
        self._plot_returns: bool = self.config['plot_returns']
        self._plot_value_functions: bool = self.config['plot_value_functions']
        self._returns: list[float] = []
        self._value_functions_sum: list[float] = []
        self.action_values: ActionValueFunction = ActionValueFunction(env=self.env, init_value=self.config['value_function_init'])
        self.policy: Policy = create_epsilon_soft_policy(action_value_function=self.action_values, epsilon=self.epsilon)
        self.model: DeterministicEnvModel = DeterministicEnvModel()

    def clear(self) -> None:
        self.env.clear()
        self._returns = []
        self._total_episodes = 0
        self._value_functions_sum = []
        self.epsilon = self.config['epsilon']
        self.action_values = ActionValueFunction(env=self.env, init_value=self.config['value_function_init'])
        self.policy = create_epsilon_soft_policy(action_value_function=self.action_values, epsilon=self.epsilon)
        self.model = DeterministicEnvModel()

    def run(self) -> None:
        for _ in range(self._iterations):
            self.run_iteration()
        if self._plot_returns or self._plot_value_functions:
            self.render()

    def render(self) -> None:
        if self._plot_value_functions:
            plot_value_function_sums(self.algo_name, self._value_functions_sum)
        if self._plot_returns:
            plot_returns(self.algo_name, self._returns)

    def run_iteration(self) -> None:
        self.run_episode()
        self.policy = self.infer_epsilon_soft_policy()
        self._value_functions_sum.append(self.action_values.sum())

    def run_episode(self) -> None:
        """
            TODO: Start an episode following the agent's policy until a terminal state (terminated) or timeout (truncated) is reached.
            Resets the environment to get the initial state. Select actions based on an epsilon-greedy policy.
            Update action values and the environment model after each step.
            After each step, perform additional planning steps using the learned model, updating the action values.
            At the end, append the total return of the episode to the returns list for plotting.
        """
        raise NotImplementedError()

    def run_planning_loop(self):
        """
            TODO: Run the planning loop by sampling a transition from the model and updating the action value function.
        """
        raise NotImplementedError()

    def update_action_value(self, state, action, reward, new_state) -> None:
        """
            TODO: Update the action value function using the Q-Learning update rule.
        """
        raise NotImplementedError()

    def calculate_action_value(self, s: State, a: Action, r: float, new_s: State) -> float:
        """
            TODO: Calculate the action value using the Q-Learning update rule.
        """
        raise NotImplementedError()

    def q_learning_error(self, s: State, a: Action, r: float, new_s: State) -> float:
        """
            TODO: Calculate the Q-Learning-Error.
            Remember the edge case when the next_state is terminal.
        """
        raise NotImplementedError()

    def infer_epsilon_soft_policy(self) -> Policy:
        """
            TODO: Create a new epsilon soft policy based on the current action values.
        """
        raise NotImplementedError()
