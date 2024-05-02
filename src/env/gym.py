import numpy as np

from src.env.action import Action
from src.env.state import State


class Gym:

    def __init__(self, gridworld, horizon=200):
        self._gridworld = gridworld
        self._agent_state = None
        self._rng = np.random.default_rng(seed=1)
        self._episode_threshold: int = horizon
        self._step_counter: int = 0

    def step(self, action: Action) -> (State, float, bool, bool):
        new_state, reward, terminated = self.plan_step(self._agent_state, action)
        self._agent_state = new_state
        self._step_counter += 1
        truncated = self._step_counter >= self._episode_threshold
        return new_state, reward, terminated, truncated

    def plan_step(self, state: State, action: Action) -> (State, float, bool):
        """
            ENVIRONMENT KNOWLEDGE METHOD
            This method only simulates a step and doesn't update the agent's position.
            That has to be done manually afterward.
            The reason for this is to be able to use this method for planning only.
        """
        if state.is_goal:  # We can't perform any action in the goal-state
            return state, 0, True

        if state.is_wall:
            raise ValueError("AgentState can't be wall! State:" + state.__str__())

        new_state = self._gridworld.get_new_state(state, action)

        reward_per_step = -1

        return new_state, reward_per_step, new_state.is_goal

    def clear(self) -> None:
        print("Clear!")
        self.reset()

    def reset(self) -> State:
        if self.gridworld.start_state is None:
            self._agent_state = self.get_random_start_state()
        else:
            self._agent_state = self.gridworld.start_state
        self._step_counter = 0
        return self._agent_state

    def get_random_start_state(self) -> State:
        return self._rng.choice(self.states)

    @property
    def states(self):
        """
        This method returns only valid 'states', that means it doesn't return walls.
        """
        return [v for k, v in self._gridworld.states.items() if not v.is_wall]

    @property
    def actions(self):
        return self._gridworld.actions

    @property
    def gridworld(self):
        return self._gridworld

    @property
    def start_state(self):
        return self._gridworld.start_state

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
