from abc import ABC, abstractmethod

from src.board.visualization.console_printer import pretty_print_to_console


class IterationStrategy(ABC):

    def __init__(self, board_layout):
        self._board_layout = board_layout
        self._state_values = None
        self.reset_rewards()

    def reset_rewards(self):
        self._state_values = self.get_init_rewards()

    def get_init_rewards(self):
        return [['0.000' for _ in range(self._board_layout.cols)] for _ in range(self._board_layout.rows)]

    def run_iteration(self):
        self.run_iteration_impl()
        pretty_print_to_console(self._state_values)

    @abstractmethod
    def run_iteration_impl(self):
        pass

    def state_values(self):
        return self._state_values

    @abstractmethod
    def action_value(self, state, action):
        pass

    @abstractmethod
    def state_value(self, state):
        pass
