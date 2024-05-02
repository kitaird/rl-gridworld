from abc import ABC, abstractmethod

import numpy as np
import configparser

from src.env.action import Action
from src.env.gym import Gym
from src.env.state import State
from src.visualization.plotter import Plotter, plot_value_deltas


class IterationStrategy(ABC):

    def __init__(self, agent_name: str, env: Gym, use_state_values: bool = True):
        self._config = configparser.ConfigParser()
        self._config.read('resources/config.ini')
        self._plotter: Plotter = Plotter(env.cols, env.rows)
        self._rng = np.random.default_rng(seed=1)
        self._agent_name: str = agent_name
        self._env: Gym = env
        self._policy: dict[State, Action] = self._random_init_policy()
        self._get_iteration_size: int = self._config.getint(self._agent_name, 'iterations')
        self._discount_factor: float = self._config.getfloat(self._agent_name, 'discount_factor')
        self._deltas: [float] = []
        self._use_state_values: bool = use_state_values
        self._should_plot: bool = self._config.getboolean(self._agent_name, 'should_plot')

    def run_iterations(self) -> None:
        for _ in range(self._get_iteration_size):
            self.run_iteration_impl()
        if self.should_plot:
            self.render()

    @abstractmethod
    def run_iteration_impl(self) -> None:
        pass

    def clear(self) -> None:
        self._env.clear()
        self._deltas = []
        self._policy = self._random_init_policy()

    def render(self) -> None:
        plot_value_deltas(self.deltas)

    def _random_init_policy(self) -> dict[State, Action]:
        return {state: self.rng.choice(self.env.actions) for state in self.env.states}

    def init_zero_state_values(self) -> dict[State, float]:
        init_state_values = {}
        for state in self.env.states:
            init_state_values[state.clone()] = 0.0
        return init_state_values

    def init_zero_action_values(self) -> dict[State, dict[Action, float]]:
        init_state_values = {}
        for state in self.env.states:
            init_state_values[state.clone()] = {action: 0.0 for action in self.env.actions}
        return init_state_values

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def env(self) -> Gym:
        return self._env

    @property
    def policy(self) -> dict[State, Action]:
        return self._policy

    @policy.setter
    def policy(self, policy: dict[State, Action]) -> None:
        self._policy = policy

    @property
    def discount_factor(self) -> float:
        return self._discount_factor

    @property
    def deltas(self):
        return self._deltas

    @property
    def use_state_values(self) -> bool:
        return self._use_state_values

    @property
    def should_plot(self) -> bool:
        return self._should_plot

    @property
    def rng(self) -> np.random.Generator:
        return self._rng
