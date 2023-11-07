class State:

    def __init__(self, row, col, is_goal=False, is_wall=False):
        self._row = row
        self._col = col
        self._is_goal = is_goal
        self._is_wall = is_wall

    @property
    def row(self):
        return self._row

    @property
    def col(self):
        return self._col

    @property
    def is_wall(self):
        return self._is_wall

    @property
    def is_goal(self):
        return self._is_goal

    def apply(self, action):
        return (self._row + action.value.row,
                self._col + action.value.col)

    def clone(self):
        return State(self._row, self._col, self._is_goal, self._is_wall)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        msg = "{"
        position = "row: " + self._row.__str__() + " , col: " + self._col.__str__()
        return msg + position + "}"

    def __hash__(self):
        return hash((self._row, self._col))

    def __eq__(self, other):
        return (self._row, self._col) == (other.row, other.col)

    def __ne__(self, other):
        return self != other
