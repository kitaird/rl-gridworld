import numpy as np

from src.env.actions import Actions
from src.env.state import State


class Environment:

    def __init__(self, init_data):
        self._init_data = init_data
        self._rows = len(init_data)
        self._cols = len(init_data[0])
        self._actions_dim = len(Actions)
        self._actions = Actions
        self._agent_state = None
        self._states = self.init_states()
        self._allowed_actions = self.get_allowed_actions()

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
    def init_data(self):
        return self._init_data

    @property
    def actions(self):
        return self._actions

    @property
    def actions_dim(self):
        return self._actions_dim

    @property
    def allowed_actions(self):
        return self._allowed_actions

    @property
    def agent_state(self):
        return self._agent_state

    @agent_state.setter
    def agent_state(self, state):
        self._agent_state = state

    def get_random_start_state(self):
        start_fields = list(self.allowed_actions.keys())
        start_index = np.random.choice(len(start_fields))
        return start_fields[start_index]

    def init_states(self):
        states = {}
        for col in range(self._cols):
            for row in range(self._rows):
                val = self._init_data[row][col]
                state = create_state(row, col, val)
                states[(row, col)] = state
        return states

    def get_new_state(self, state, action):
        new_state_coords = state.apply(action)
        if self.is_outside_bounds(new_state_coords):
            return state
        new_state = self._states[new_state_coords]
        if new_state.is_wall:
            return state
        return new_state

    def get_new_state_and_reward(self, state, action):
        if state.is_goal:
            return state, 0

        if state.is_wall:
            raise ValueError("State can't be wall!")

        new_state = self.get_new_state(state, action)

        reward_any_step = -1

        return new_state, reward_any_step

    def is_outside_bounds(self, state_coords):
        outside_rows = state_coords[0] < 0 or state_coords[0] >= self.rows
        outside_cols = state_coords[1] < 0 or state_coords[1] >= self.cols
        return outside_rows or outside_cols

    def get_allowed_actions(self):
        allowed_actions = {}
        for state in self.states.values():
            if not state.is_wall:
                allowed_actions[state.clone()] = self.get_allowed_actions_for(state)
        return allowed_actions

    def get_allowed_actions_for(self, state):
        possible_actions = []
        for action in self.actions:
            try:
                next_state_pos = state.apply(action)
                next_state = self.states[next_state_pos]
                is_bound = next_state.is_wall
            except:
                is_bound = True
            if not is_bound:
                possible_actions.append(action)
        return possible_actions


def create_state(row, col, val):
    if val == 'g':
        return State(row, col, is_goal=True)
    elif val == 1:
        return State(row, col, is_wall=True)
    return State(row, col)
