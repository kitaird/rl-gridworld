
class BoardState:

    def __init__(self, board_layout) -> None:
        self._rows = len(board_layout)
        self._cols = len(board_layout[0])
        self._layout = board_layout

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def layout(self):
        return self._layout

    def get_field(self, cell):
        return self._layout[cell.row][cell.col]

    def is_outside_bounds(self, cell):
        outside_rows = cell.row < 0 or cell.row >= self.rows
        outside_cols = cell.col < 0 or cell.col >= self.cols
        return outside_rows or outside_cols


class State:

    def __init__(self, row, col):
        self._row = row
        self._col = col

    @property
    def row(self):
        return self._row

    @property
    def col(self):
        return self._col

