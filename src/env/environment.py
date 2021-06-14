from src.env.actions import Actions


class Environment:

    def __init__(self, board_layout):
        self._rows = len(board_layout)
        self._cols = len(board_layout[0])
        self._layout = board_layout
        self._actions_dim = len(Actions)
        self._actions = Actions

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def layout(self):
        return self._layout

    @property
    def actions(self):
        return self._actions

    @property
    def actions_dim(self):
        return self._actions_dim

    def is_goal(self, cell):
        return self.get_field(cell) == 'g'

    def is_wall(self, cell):
        return self.get_field(cell) == 1

    def get_field(self, cell):
        return self.layout[cell.row][cell.col]

    def is_outside_bounds(self, cell):
        outside_rows = cell.row < 0 or cell.row >= self.rows
        outside_cols = cell.col < 0 or cell.col >= self.cols
        return outside_rows or outside_cols
