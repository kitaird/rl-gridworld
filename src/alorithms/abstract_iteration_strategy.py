from abc import ABC, abstractmethod

from src.board.board_service import BoardCalculationService
from src.board.visualization.console_printer import pretty_print_to_console


class IterationStrategy(ABC):

    def __init__(self, board_layout):
        self._board_layout = board_layout
        self._board_service = BoardCalculationService(self._board_layout)
        self._state_values = self._board_service.init_state_values()

    def reset_rewards(self):
        self._state_values = self._board_service.init_state_values()

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
