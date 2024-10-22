from typing import  Optional

import numpy as np

from src.env.action import Action
from src.env.gridworld import Gridworld
from src.env.state import State


class Gym:

    def __init__(self, gridworld: Gridworld):
        self._gridworld: Gridworld = gridworld
        self._start_state: State = self._gridworld.start_state
        self._agent_state: Optional[State] = None
        self._rng = np.random.default_rng(seed=1)
        self.episode_threshold: int = gridworld.horizon
        self._step_counter: int = 0

    def step(self, action: Action) -> (State, float, bool, bool):
        new_state, reward, terminated = self.plan_step(self._agent_state, action)
        self._agent_state = new_state
        self._step_counter += 1
        truncated = self._step_counter >= self.episode_threshold
        return new_state, reward, terminated, truncated

    def plan_step(self, state: State, action: Action) -> (State, float, bool):
        """
            ENVIRONMENT KNOWLEDGE METHOD
            This method only simulates a step and doesn't update the agent's position.
            That has to be done manually afterward.
            The reason for this is to be able to use this method for planning only.
        """
        assert not state.terminal, ValueError("AgentState can't be terminal! State:" + state.__str__())
        assert not state.wall, ValueError("AgentState can't be wall! State:" + state.__str__())
        new_state = self._gridworld.get_new_state(state, action)
        reward = self._gridworld.reward_per_step + new_state.reward
        return new_state, reward, new_state.terminal

    def clear(self) -> None:
        print("Clear!")
        self.reset()

    def reset(self) -> State:
        self._agent_state = self._start_state if self._start_state is not None else self.get_random_start_state()
        self._step_counter = 0
        return self._agent_state

    def get_random_start_state(self) -> State:
        return self._rng.choice([*self.valid_states])

    @property
    def valid_states(self) -> list[State]:
        """
        This method returns only valid states, that means it doesn't return walls or terminal states.
        """
        return [state for state in self._gridworld.states.values() if not (state.wall or state.terminal)]

    @property
    def actions(self):
        return self._gridworld.actions

    @property
    def start_state(self) -> State:
        return self._start_state

    # Property required for visualization
    @property
    def all_states(self) -> dict[tuple[int, int], State]:
        return self._gridworld.states

    @property
    def rows(self) -> int:
        return self._gridworld.rows

    @property
    def cols(self) -> int:
        return self._gridworld.cols
