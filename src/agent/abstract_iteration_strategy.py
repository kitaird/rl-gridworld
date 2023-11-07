import copy
from abc import ABC, abstractmethod

import numpy as np


class IterationStrategy(ABC):

    def __init__(self, agent_name, env):
        self._agent_name = agent_name
        self._env = env
        self._policy = self._random_init_policy()
        self._get_iteration_size = 10
        self._discount_factor = 0.9

    def run_iterations(self):
        for _ in range(self._get_iteration_size):
            self.run_iteration_impl()
        self._env.render()

    @abstractmethod
    def run_iteration_impl(self):
        pass

    def reset(self):
        print("Reset!")
        self._env.reset()
        self._policy = self._random_init_policy()

    def _random_init_policy(self):
        random_init_policy = {}
        for state in self._env.states:
            state_copy = state.clone()
            random_init_policy[state_copy] = np.random.choice(self._env.actions)
        return random_init_policy

    def clone_state_values(self):
        cloned_state_values = {}
        for state, value in self._env.state_values.items():
            cloned_state_values[state.clone()] = copy.copy(value)
        return cloned_state_values

    def get_greedy_action_for_state(self, state):
        best_val = float('-inf')
        best_action = None
        for action in self._env.actions:
            next_state, _ = self._env.simulate_step(state, action)
            val = self._env.state_values[next_state]
            if val > best_val:
                best_val = val
                best_action = action
        return best_action

    @property
    def state_values(self):
        return self._env.state_values

    @property
    def agent_name(self):
        return self._agent_name

    @property
    def env(self):
        return self._env

    @property
    def policy(self):
        return self._policy

    @property
    def discount_factor(self):
        return self._discount_factor
