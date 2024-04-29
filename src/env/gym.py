import numpy as np

from src.env.action import Action
from src.env.state import State
from src.visualization.plotter import Plotter


class Gym:

    def __init__(self, gridworld):
        self._gridworld = gridworld
        self._agent_state = None
        self._state_values = self.init_zero_state_values()
        self._plotter = Plotter(self)
        self._deltas = []
        self._rng = np.random.default_rng(seed=1)

    def get_random_start_state(self) -> State:
        return self._rng.choice(self.states)

    def step(self, state: State, action: Action) -> (State, float):
        new_state, reward = self.simulate_step(state, action)
        self.agent_state = new_state
        return new_state, reward

    """
        ENVIRONMENT KNOWLEDGE METHOD
        This method only simulates a step and doesn't update the agent's position.
        That has to be done manually afterwards. 
        The reason for this is to be able to use this method for planning only.
    """
    def simulate_step(self, state: State, action: Action) -> (State, float):
        if state.is_goal:
            return state, 0

        if state.is_wall:
            raise ValueError("AgentState can't be wall! State:" + state.__str__())

        new_state = self._gridworld.get_new_state(state, action)

        reward_per_step = -1

        return new_state, reward_per_step

    def reset(self) -> None:
        self._state_values = self.init_zero_state_values()
        self._deltas = []
        self.render()

    def render(self) -> None:
        self._plotter.plot_state_value_deltas()
        self._plotter.pretty_print_to_console()

    def init_zero_state_values(self) -> {State, float}:
        init_state_values = {}
        for state in self.states:
            state_copy = state.clone()
            init_state_values[state_copy] = 0.0
        return init_state_values

    """
    This method returns only valid 'states', that means it doesn't return walls.
    """
    @property
    def states(self):
        return [v for k, v in self._gridworld.states.items() if not v.is_wall]

    @property
    def actions(self):
        return self._gridworld.actions

    @property
    def agent_state(self):
        return self._agent_state

    @agent_state.setter
    def agent_state(self, state: State):
        self._agent_state = state

    @property
    def state_values(self):
        return self._state_values

    @state_values.setter
    def state_values(self, new_state_values: {State, float}):
        self._state_values = new_state_values

    @property
    def deltas(self):
        return self._deltas

    @property
    def gridworld(self):
        return self._gridworld

    # Properties needed for plotting
    @property
    def init_data(self):
        return self._gridworld.init_data

    @property
    def rows(self):
        return self._gridworld.rows

    @property
    def cols(self):
        return self._gridworld.cols
