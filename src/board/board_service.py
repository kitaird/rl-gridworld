from src.board.board_state import BoardCell


def add_cells(cell1, cell2):
    return BoardCell(cell1.row + cell2.row, cell1.col + cell2.col)


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
