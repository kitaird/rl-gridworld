import random
from typing import Dict, Tuple

from src.env.action import Action
from src.env.state import State


class DeterministicEnvModel:
    """
    A deterministic environment model that maintains state transitions and rewards.
    Used for model-based reinforcement learning algorithms to store and sample from
    previously experienced state-action-reward-next_state tuples.
    """

    def __init__(self) -> None:
        self._transitions: Dict[Tuple[State, Action], Tuple[State, float]] = {}

    def update(self, state: State, action: Action, reward: float, next_state: State) -> None:
        """
                Update the model with a new state transition.

                Args:
                    state: The current state
                    action: The action taken
                    reward: The reward received
                    next_state: The resulting state
                """
        self._transitions[(state, action)] = (next_state, reward)

    def sample_transition(self) -> Tuple[State, Action]:
        """Sample a random state-action pair from the model."""
        return random.choice(list(self._transitions.keys()))

    def step(self, state: State, action: Action) -> tuple[State, float]:
        """
            Get the next_state and reward for a given state-action pair.
        """
        return self._transitions.get((state, action))
