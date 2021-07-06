import numpy as np

from src.agent.abstract_iteration_strategy import IterationStrategy


class TdIterationStrategy(IterationStrategy):
    """
        This is the Temporal Difference learning strategy
        There is no domain knowledge available
        The state value map should be updated during episode run
    """

    def __init__(self, env):
        super().__init__(env)
        self._step_size = 0.1

    def get_iteration_size(self):
        return 100

    def discount_factor(self):
        return 0.75

    def get_agent_name(self):
        return "Temporal difference learning"

    def run_iteration_impl(self):
        pass

    def new_state_value(self, state, next_state, reward):
        pass

    def play_game(self):
        pass
