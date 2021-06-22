from abc import ABC, abstractmethod

import numpy as np

from src.env.actions import Actions
from src.visualization.plotter import Plotter


class IterationStrategy(ABC):

    def __init__(self, env):
        self._env = env
        self._state_values = self.init_zero_state_values()
        self._policy = self.random_init_policy()
        self._deltas = []
        self._plotter = Plotter(self)

    @property
    def env(self):
        return self._env

    @property
    def state_values(self):
        return self._state_values

    @property
    def policy(self):
        return self._policy

    @property
    def deltas(self):
        return self._deltas

    def random_init_policy(self):
        random_init_policy = {}
        for state in self.env.states.values():
            if not state.is_wall:
                state_copy = state.clone()
                random_init_policy[state_copy] = np.random.choice(Actions)
        return random_init_policy

    def reset(self):
        print("Reset!")
        self._state_values = self.init_zero_state_values()
        self._policy = self.random_init_policy()
        self._deltas = []
        self._plotter.plot_state_value_deltas()
        self._plotter.pretty_print_to_console()

    def init_zero_state_values(self):
        return self._init_state_values(lambda: 0.0)

    def init_random_state_values(self):
        return self._init_state_values(lambda: np.random.random())

    def _init_state_values(self, value_provider):
        init_state_values = {}
        for state in self.env.states.values():
            if not state.is_wall:
                state_copy = state.clone()
                init_state_values[state_copy] = value_provider()
        return init_state_values

    def run_iterations(self):
        for _ in range(self.get_iteration_size()):
            self.run_iteration_impl()
        self._plotter.plot_state_value_deltas()
        self._plotter.pretty_print_to_console()

    def get_action_for_state(self, state):
        allowed_actions = self.env.allowed_actions[state]
        best_val = float('-inf')
        best_action = None
        for action in allowed_actions:
            next_state, _ = self.env.get_new_state_and_reward(state, action)
            val = self._state_values[next_state]
            if val > best_val:
                best_val = val
                best_action = action
        return best_action

    @abstractmethod
    def get_agent_name(self):
        pass

    @abstractmethod
    def get_iteration_size(self):
        pass

    @abstractmethod
    def discount_factor(self):
        pass

    @abstractmethod
    def run_iteration_impl(self):
        pass
