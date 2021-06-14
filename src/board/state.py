class State:

    def __init__(self, row, col, val=None):
        self._row = row
        self._col = col
        self._val = val

    @property
    def row(self):
        return self._row

    @property
    def col(self):
        return self._col

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        self._val = value

    def has_val(self):
        return self._val is not None

    def apply(self, action):
        return State(self._row + action.value.row,
                     self._col + action.value.col)

    def __str__(self):
        msg = "{"
        position = "row: "+self._row.__str__()+" , col: " + self._col.__str__()
        value = " , val: " + self._val.__str__() if self.has_val else ""
        return msg + position + value + "}"
