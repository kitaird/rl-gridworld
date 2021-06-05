from abc import ABC, abstractmethod

from src.board_state import BoardState
from src.board_utils import BoardCalculationService


class IterationStrategy(ABC):

    def __init__(self, board_layout) -> None:
        super().__init__()
        self._board_layout = BoardState(board_layout)
        self._board_service = BoardCalculationService(self._board_layout)
        self._last_rewards = self._board_service.init_rewards()

    @abstractmethod
    def run_iteration(self):
        pass

