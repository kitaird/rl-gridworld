from src.board_state import BoardCell


def add_cells(cell1, cell2):
    return BoardCell(cell1.row + cell2.row, cell1.col + cell2.col)


class BoardCalculationService:

    def __init__(self, board_layout) -> None:
        super().__init__()
        self._board_layout = board_layout

    def init_rewards(self):
        return [[0.0 for _ in range(self._board_layout.cols)] for _ in range(self._board_layout.rows)]

    def is_goal(self, cell):
        return self._board_layout.layout[cell.row][cell.col] == 'g'

    def is_wall(self, cell):
        return self._board_layout.layout[cell.row][cell.col] == 1

    def is_outside_bounds(self,  cell):
        return cell.row < 0 or cell.row >= self._board_layout.rows or cell.col < 0 or cell.col >= self._board_layout.cols

