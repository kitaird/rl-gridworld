from src.alorithms.dp_iteration_strategy import DpIterationStrategy
from src.board.actions import Actions
from src.board.board_service import add_cells
from src.board.board_state import BoardCell
from src.board.drl_board import DrlBoard
import numpy as np

data = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 'g']]

cols = 9
rows = 6

board = DrlBoard(rows, cols)
strategy = DpIterationStrategy(data)


def fill_field(row, col, value):
    board[row][col] = value


def init_board():
    board.title = "Grid world game!"
    board.cell_size = 90
    board.cell_color = "white"


def reward_for_cell(btn, row, col):
    print(board[row][col])


def show_loss():
    show_grid()
    last_rewards = strategy.show_loss()
    for row in range(len(last_rewards)):
        for col in range(len(last_rewards[0])):
            val = last_rewards[row][col]
            if val == '-' or val == '0':
                pass
            else:
                fill_field(row, col, val)


def show_gradient():
    show_grid()
    last_rewards = strategy.show_loss()
    for row in range(len(last_rewards)):
        for col in range(len(last_rewards[0])):
            left = get_left(last_rewards, BoardCell(row, col))
            right = get_right(last_rewards, BoardCell(row, col))
            up = get_up(last_rewards, BoardCell(row, col))
            down = get_down(last_rewards, BoardCell(row, col))
            arr = np.array([left, right, up, down])
            gradient = np.sort(arr)[0]
            leftish = gradient == left
            rightish = gradient == right
            upish = gradient == up
            downish = gradient == down

            if leftish and rightish and upish and downish:
                pass
                # do nothing
            elif leftish:
                fill_field(row, col, 'left')
            elif rightish:
                fill_field(row, col, 'right')
            elif upish:
                fill_field(row, col, 'up')
            elif downish:
                fill_field(row, col, 'down')


def get_left(rewards, cell):
    return get_reward(rewards, cell, add_cells(cell, Actions.left.value))


def get_right(rewards, cell):
    return get_reward(rewards, cell, add_cells(cell, Actions.right.value))


def get_up(rewards, cell):
    return get_reward(rewards, cell, add_cells(cell, Actions.up.value))


def get_down(rewards, cell):
    return get_reward(rewards, cell, add_cells(cell, Actions.down.value))


def get_reward(rewards, cell, next_cell):
    cell_reward = rewards[cell.row][cell.col]
    if next_cell.row >= len(rewards) or next_cell.row < 0 or next_cell.col >= len(rewards[0]) or next_cell.col < 0:
        next_cell = cell
    next_cell_reward = rewards[next_cell.row][next_cell.col]
    if next_cell_reward == '-' or cell_reward == '-':
        return 0
    return float(cell_reward) - float(next_cell_reward)


def show_grid():
    board.load(data)


init_board()
board.on_mouse_click = reward_for_cell
board.iterate_command = strategy.run_iteration
board.show_loss_command = show_loss
board.show_gradient_command = show_gradient
board.show_grid_command = show_grid
board.load(data)
board.show()

# move = -1 reward
# move into wall -1 (stays in same spot)
# move out of grid -1 (stays in same spot)
# move into goal 0
# move into goal -> no more jumping

# Starting with 0 reward for all fields..
