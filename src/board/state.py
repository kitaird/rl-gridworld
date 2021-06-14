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

    def apply(self, action):
        return State(self._row + action.value.row,
                     self._col + action.value.col)