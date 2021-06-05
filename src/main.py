from src.dp_iteration_strategy import DpIterationStrategy
from src.drl_board import DrlBoard

data = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 'g']]

cols = 9
rows = 6

board = DrlBoard(rows, cols)


def fill_field(row, col, value):
    board[row][col] = value


def init_board():
    board.title = "Grid world game!"
    board.cell_size = 90
    board.cell_color = "white"


def reward_for_cell(btn, row, col):
    print(board[row][col])


init_board()
board.on_mouse_click = reward_for_cell
board.iterate_command = DpIterationStrategy(data).run_iteration
board.load(data)
board.show()

# move = -1 reward
# move into wall -1 (stays in same spot)
# move out of grid -1 (stays in same spot)
# move into goal 0
# move into goal -> no more jumping

# Starting with 0 reward for all fields..
