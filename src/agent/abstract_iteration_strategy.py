from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

import numpy as np
from tabulate import tabulate

from src.env.actions import Actions
from src.env.state import State


class IterationStrategy(ABC):

    def __init__(self, env):
        self._env = env
        self._state_values = self.init_state_values()
        self._allowed_actions = self.get_allowed_actions()
        self._policy = self.random_init_policy()
        self._deltas = []

    def random_init_policy(self):
        random_init_policy = {}
        for state in self._env.states.values():
            if not state.is_wall:
                state_copy = state.clone()
                random_init_policy[state_copy] = np.random.choice(Actions)
        return random_init_policy

    def reset(self):
        print("Reset!")
        self._state_values = self.init_state_values()
        self._policy = self.random_init_policy()
        self._deltas = []
        self.plot_state_value_deltas()
        self.pretty_print_to_console()

    def init_state_values(self):
        init_state_values = {}
        for state in self._env.states.values():
            if not state.is_wall:
                state_copy = state.clone()
                init_state_values[state_copy] = 0 if state_copy.is_goal else np.random.random()
        return init_state_values

    def run_iteration(self):
        self.run_iteration_impl()
        self.plot_state_value_deltas()
        self.pretty_print_to_console()

    def get_allowed_actions(self):
        allowed_actions = {}
        for state in self._env.states.values():
            if not state.is_wall:
                allowed_actions[state.clone()] = self.get_allowed_actions_for(state)
        return allowed_actions

    def get_allowed_actions_for(self, state):
        possible_actions = []
        for action in self._env.actions:
            try:
                next_state_pos = state.apply(action)
                next_state = self._env.states[next_state_pos]
                is_bound = next_state.is_wall
            except:
                is_bound = True
            if not is_bound:
                possible_actions.append(action)
        return possible_actions

    def pretty_print_to_console(self):
        print("New rewards!")
        values_to_print = [[self.get_printable_state(r, c) for c in range(self._env.cols)] for r in
                           range(self._env.rows)]
        print(tabulate(values_to_print))

    def get_printable_state(self, row, col):
        state_to_find = State(row, col)
        state_value = self._state_values.get(state_to_find)
        return "{:1.3f}".format(state_value) if state_value is not None else 'WALL'

    def plot_state_value_deltas(self):
        plt.plot(self._deltas)
        plt.xlabel("Episodes")
        plt.ylabel("State-Value Delta (highest delta per episode")
        plt.title("State-Value Convergence")
        plt.show()

    @property
    def policy(self):
        return self._policy

    @property
    def state_values(self):
        return self._state_values

    @abstractmethod
    def discount_factor(self):
        pass

    @abstractmethod
    def run_iteration_impl(self):
        pass
