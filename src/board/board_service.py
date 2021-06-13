from src.board.board_state import State


def get_new_state(cell, action):
    return State(cell.row + action.value.row, cell.col + action.value.col)


class BoardCalculationService:

    def __init__(self, board_layout):
        self._board_layout = board_layout

    def init_state_values(self):
        return [['0.000' for _ in range(self._board_layout.cols)] for _ in range(self._board_layout.rows)]

    def is_goal(self, cell):
        return self._board_layout.get_field(cell) == 'g'

    def is_wall(self, cell):
        return self._board_layout.get_field(cell) == 1

    def is_outside_bounds(self, cell):
        return self._board_layout.is_outside_bounds(cell)
