from pathlib import Path
import numpy as np
import yaml

from src.env.action import Action, cardinal_moves, queens_moves
from src.env.state import State


class Gridworld:

    def __init__(self, config_path=None):
        with open(Path(__file__).parent / 'gridworld-config.yml' if config_path is None else config_path) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)['gridworld']
        _layout = self._config['Layout']
        _rewards = self._config['Rewards']
        assert np.array(_layout).shape == np.array(_rewards).shape
        self._horizon: int = self._config['Horizon']
        self._actions: list[Action] = cardinal_moves if self._config['Action_space'] == 'cardinal_moves' else queens_moves
        self._start_state: State = None
        self._states: dict[tuple[int, int], State] = self._init_gridworld(_layout, _rewards)
        self._reward_per_step: float = self._config['Reward_per_step']
        self._rows: int = np.array(_layout).shape[0]
        self._cols: int = np.array(_layout).shape[1]

    def _init_gridworld(self, layout, rewards) -> dict[tuple[int, int], State]:
        states = {}
        for row in range(len(layout)):
            assert len(layout[row]) == len(layout[0]), 'All rows must have the same number of columns'
            for col in range(len(layout[row])):
                assert layout[row][col] in ['.', 'w', 't', 's'], 'Invalid character in map'
                is_wall = layout[row][col] == 'w'
                is_terminal = layout[row][col] == 't'
                rew_val = rewards[row][col]
                reward = rew_val if isinstance(rew_val, int) or isinstance(rew_val, float) else 0
                states[(row, col)] = State(row, col, reward, is_wall, is_terminal)
                if layout[row][col] == 's':
                    self._start_state = states[(row, col)]
        return states

    def get_new_state(self, state, action) -> State:
        new_state_coords = state.apply(action)
        if self._is_outside_bounds(new_state_coords):
            return state
        new_state = self._states[new_state_coords]
        if new_state.wall:
            return state
        return new_state

    def _is_outside_bounds(self, state_coords: tuple[int, int]) -> bool:
        outside_rows = state_coords[0] < 0 or state_coords[0] >= self.rows
        outside_cols = state_coords[1] < 0 or state_coords[1] >= self.cols
        return outside_rows or outside_cols

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def states(self) -> dict[tuple[int, int], State]:
        return self._states

    @property
    def actions(self):
        return self._actions

    @property
    def start_state(self) -> State:
        return self._start_state

    @property
    def reward_per_step(self) -> float:
        return self._reward_per_step
