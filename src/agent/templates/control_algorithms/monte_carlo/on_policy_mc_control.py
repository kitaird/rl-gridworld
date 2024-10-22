from pathlib import Path
from typing import Optional

import yaml

from src.agent.common.policies import Policy, create_epsilon_soft_policy
from src.agent.common.value_functions import ActionValueFunction
from src.env.action import Action
from src.env.gym import Gym
from src.env.state import State
from src.visualization.plotter import plot_returns, plot_value_function_sums


class OnPolicyMcControl:
    """
        This is the On-Policy Monte Carlo Control algorithm.

        Collect trajectories and update the policy based on the collected experience.

        For reference, see Sutton & Barto, Reinforcement Learning: An Introduction, 2018, p. 97, chapter 5.3, Monte Carlo Control and p. 100, chapter 5.4, Monte Carlo Control without Exploring Starts.
    """

    def __init__(self, env):
        self.algo_name: str = "On-Policy MC Control"
        with open(Path(__file__).parent.parent / 'algorithms-config.yml') as f:
            self.config = yaml.safe_load(f)[self.algo_name]
        self.env: Gym = env
        self.discount_factor: float = self.config['discount_factor']
        self.step_size: Optional[float] = self.config['step_size']
        self.epsilon: float = self.config['epsilon']
        self.epsilon_decay: float = self.config['epsilon_decay']
        self._total_episodes: int = 0
        self._num_of_episodes_to_collect: int = self.config['num_of_episodes_to_collect']
        self._use_every_visit: bool = self.config['use_every_visit']
        self._iterations: int = self.config['iterations']
        self._plot_returns: bool = self.config['plot_returns']
        self._plot_value_functions: bool = self.config['plot_value_functions']
        self._returns: list[float] = []
        self._value_functions_sum: list[float] = []
        self.action_values: ActionValueFunction = ActionValueFunction(env=self.env, init_value=self.config['value_function_init'])
        self.policy: Policy = create_epsilon_soft_policy(action_value_function=self.action_values, epsilon=self.epsilon)
        self._q_returns: dict[State, dict[Action, list[float]]] = self._init_q_returns()

    def _init_q_returns(self) -> dict[State, dict[Action, list[float]]]:
        return {s.clone(): {a: [] for a in self.env.actions} for s in self.env.valid_states}

    def clear(self) -> None:
        self.env.clear()
        self._returns = []
        self._total_episodes = 0
        self._value_functions_sum = []
        self.epsilon = self.config['epsilon']
        self._q_returns = self._init_q_returns()
        self.action_values = ActionValueFunction(env=self.env, init_value=self.config['value_function_init'])
        self.policy = create_epsilon_soft_policy(action_value_function=self.action_values, epsilon=self.epsilon)

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
        self.policy_evaluation()
        self.policy = self.infer_epsilon_soft_policy()
        self._value_functions_sum.append(self.action_values.sum())

    def policy_evaluation(self) -> None:
        """
            TODO: Use the generate_trajectory method to collect self._num_of_episodes_to_collect experience.
            Calculate the return G for each state-action pair.
            Update the values for each seen state-action pair using the return.
            Implement the first-visit and every-visit method depending on the self._use_every_visit flag.
            At the end, append the total return of the episode to the returns list for plotting.
        """
        raise NotImplementedError()

    def compute_action_value(self, s: State, a: Action, G: float) -> float:
        """
            TODO: Compute the new action value for the given state-action pair.
            If self.step_size is None, return the true mean, else a sliding average.
        """
        raise NotImplementedError()

    def infer_epsilon_soft_policy(self) -> Policy:
        """
            TODO: Create a new epsilon soft policy based on the current action values.
        """
        raise NotImplementedError()

    def generate_trajectory(self) -> [(State, Action, float)]:
        """
            TODO: Generate a trajectory.

            While not reaching a terminal state, perform steps according to the agent's policy.
            Store the next_state, action and reward as tuples in a list.
            Let epsilon decay after each episode.

            This method returns a list of tuples
                [
                    (state_t, action_t, reward_t+1),
                    ...
                ]
        """
        raise NotImplementedError()
