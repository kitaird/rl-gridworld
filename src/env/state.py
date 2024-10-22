
class State:

    def __init__(self, row: int, col: int, reward: float = 0, is_wall: bool = False, is_terminal: bool = False):
        self._row = row
        self._col = col
        self._reward = reward
        self._wall = is_wall
        self._terminal = is_terminal

    @property
    def row(self) -> int:
        return self._row

    @property
    def col(self) -> int:
        return self._col

    @property
    def wall(self) -> bool:
        return self._wall

    @wall.setter
    def wall(self, is_wall: bool) -> None:
        self._wall = is_wall
        self.reward = 0

    @property
    def terminal(self) -> bool:
        return self._terminal

    @terminal.setter
    def terminal(self, is_terminal: bool) -> None:
        self._terminal = is_terminal

    @property
    def reward(self) -> float:
        return self._reward

    @reward.setter
    def reward(self, rew: float) -> None:
        self._reward = rew

    def apply(self, action) -> tuple[int, int]:
        return (self._row + action.value.row,
                self._col + action.value.col)

    def clone(self) -> "State":
        return State(self._row, self._col, reward=self._reward, is_wall=self._wall, is_terminal=self._terminal)

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
