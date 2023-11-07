from src.env.actions import Actions
from src.env.state import State


class Gridworld:

    def __init__(self, init_data):
        self._actions = Actions
        self._init_data = init_data
        self._rows = len(init_data)
        self._cols = len(init_data[0])
        self._states = self._init_gridworld()

    def get_new_state(self, state, action):
        new_state_coords = state.apply(action)
        if self._is_outside_bounds(new_state_coords):
            return state
        new_state = self._states[new_state_coords]
        if new_state.is_wall:
            return state
        return new_state

    def _init_gridworld(self):
        states = {}
        for col in range(self._cols):
            for row in range(self._rows):
                val = self._init_data[row][col]
                state = create_state(row, col, val)
                states[(row, col)] = state
        return states

    def _is_outside_bounds(self, state_coords):
        outside_rows = state_coords[0] < 0 or state_coords[0] >= self.rows
        outside_cols = state_coords[1] < 0 or state_coords[1] >= self.cols
        return outside_rows or outside_cols

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions

    @property
    def init_data(self):
        return self._init_data


def create_state(row, col, val):
    if val == 'g':
        return State(row, col, is_goal=True)
    elif val == 1:
        return State(row, col, is_wall=True)
    return State(row, col)
