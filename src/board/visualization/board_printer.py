import numpy as np

from src.board.actions import Actions
from src.board.board_service import add_cells
from src.board.board_state import BoardCell


def get_reward(rewards, cell, next_cell):
    cell_reward = rewards[cell.row][cell.col]
    if next_cell.row >= len(rewards) or next_cell.row < 0 or next_cell.col >= len(rewards[0]) or next_cell.col < 0:
        next_cell = cell
    next_cell_reward = rewards[next_cell.row][next_cell.col]
    if next_cell_reward == '-' or cell_reward == '-':
        return 0
    return float(cell_reward) - float(next_cell_reward)


def get_left_gradient(rewards, cell):
    return get_reward(rewards, cell, add_cells(cell, Actions.left.value))


def get_right_gradient(rewards, cell):
    return get_reward(rewards, cell, add_cells(cell, Actions.right.value))


def get_up_gradient(rewards, cell):
    return get_reward(rewards, cell, add_cells(cell, Actions.up.value))


def get_down_gradient(rewards, cell):
    return get_reward(rewards, cell, add_cells(cell, Actions.down.value))


class BoardPrinter:

    def __init__(self, drl_board):
        self._board = drl_board
        self._strategy = drl_board.strategy

    def show_gradient(self):
        self._board.show_grid()
        last_rewards = self._strategy.show_loss()
        for row in range(len(last_rewards)):
            for col in range(len(last_rewards[0])):
                left = get_left_gradient(last_rewards, BoardCell(row, col))
                right = get_right_gradient(last_rewards, BoardCell(row, col))
                up = get_up_gradient(last_rewards, BoardCell(row, col))
                down = get_down_gradient(last_rewards, BoardCell(row, col))
                arr = np.array([left, right, up, down])
                highest_gradient = np.sort(arr)[0]
                leftish = highest_gradient == left
                rightish = highest_gradient == right
                upish = highest_gradient == up
                downish = highest_gradient == down

                if leftish and rightish and upish and downish:
                    pass
                    # do nothing
                elif leftish:
                    self._board.fill_field(row, col, 'left')
                elif rightish:
                    self._board.fill_field(row, col, 'right')
                elif upish:
                    self._board.fill_field(row, col, 'up')
                elif downish:
                    self._board.fill_field(row, col, 'down')

    def show_loss(self):
        self._board.show_grid()
        last_rewards = self._strategy.show_loss()
        for row in range(len(last_rewards)):
            for col in range(len(last_rewards[0])):
                val = last_rewards[row][col]
                if val == '-' or val == '0':
                    pass
                else:
                    self._board.fill_field(row, col, val)
