from abc import ABC, abstractmethod

from src.board.board_state import BoardState
from src.board.board_service import BoardCalculationService
from src.board.visualization.console_printer import pretty_print_to_console


class IterationStrategy(ABC):

    def __init__(self, board_layout) -> None:
        super().__init__()
        self._board_layout = BoardState(board_layout)
        self._board_service = BoardCalculationService(self._board_layout)
        self._last_rewards = self._board_service.init_rewards()

    def reset_rewards(self):
        self._last_rewards = self._board_service.init_rewards()

    def run_iteration(self):
        self.run_iteration_impl()
        pretty_print_to_console(self._last_rewards)

    @abstractmethod
    def run_iteration_impl(self):
        pass

    def show_loss(self):
        return self._last_rewards
