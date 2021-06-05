
class BoardState:

    def __init__(self, board_layout) -> None:
        super().__init__()
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


class BoardCell:

    def __init__(self, row, col) -> None:
        super().__init__()
        self._row = row
        self._col = col

    @property
    def row(self):
        return self._row

    @property
    def col(self):
        return self._col

