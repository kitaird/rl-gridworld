import random

from src.env.action import Action
from src.env.state import State


class DeterministicEnvModel:

    def __init__(self):
        self.model: dict[tuple[State, Action], tuple[State, float]] = {}

    def update(self, state, action, reward, next_state) -> None:
        self.model[(state, action)] = (next_state, reward)

    def sample_transition(self) -> tuple[State, Action]:
        return random.choice(list(self.model.keys()))

    def get(self, state, action) -> tuple[State, float]:
        return self.model.get((state, action))