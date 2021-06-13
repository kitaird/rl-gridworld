import numpy as np
from src.board.actions import Actions
from src.board.board_state import State


def get_reward(state_values, cell, next_cell):
    cell_reward = state_values[cell.row][cell.col]
    if next_cell.row >= len(state_values) or next_cell.row < 0 or next_cell.col >= len(state_values[0]) or next_cell.col < 0:
        next_cell = cell
    next_cell_reward = state_values[next_cell.row][next_cell.col]
    if next_cell_reward == '-' or cell_reward == '-':
        return 0
    return float(cell_reward) - float(next_cell_reward)


def get_left_gradient(state_values, cell):
    return get_reward(state_values, cell, cell.apply(Actions.left))


def get_right_gradient(state_values, cell):
    return get_reward(state_values, cell, cell.apply(Actions.right))


def get_up_gradient(state_values, cell):
    return get_reward(state_values, cell, cell.apply(Actions.up))


def get_down_gradient(state_values, cell):
    return get_reward(state_values, cell, cell.apply(Actions.down))


def more_than_one_true(b1, b2, b3, b4):
    if b1:
        return b2 or b3 or b4
    if b2:
        return b3 or b4
    return b3 and b4


class BoardPrinter:

    def __init__(self, drl_board):
        self._board = drl_board
        self._strategy = drl_board.strategy

    def show_gradient(self):
        self._board.show_grid()
        state_values = self._strategy.state_values()
        for row in range(len(state_values)):
            for col in range(len(state_values[0])):
                left = get_left_gradient(state_values, State(row, col))
                right = get_right_gradient(state_values, State(row, col))
                up = get_up_gradient(state_values, State(row, col))
                down = get_down_gradient(state_values, State(row, col))
                # TODO if wall dont do!
                # left = self._strategy.action_value(State(row, col), Actions.left)
                # right = self._strategy.action_value(State(row, col), Actions.right)
                # up = self._strategy.action_value(State(row, col), Actions.up)
                # down = self._strategy.action_value(State(row, col), Actions.down)
                arr = np.array([left, right, up, down])
                highest_gradient = np.sort(arr)[0]
                # highest_gradient = np.sort(arr[~np.isnan(arr.astype(float))])[0]
                leftish = highest_gradient == left
                rightish = highest_gradient == right
                upish = highest_gradient == up
                downish = highest_gradient == down

                if more_than_one_true(leftish, rightish, upish, downish):
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
        state_values = self._strategy.state_values()
        for row in range(len(state_values)):
            for col in range(len(state_values[0])):
                val = state_values[row][col]
                if val == '-' or val == '0':
                    pass
                else:
                    self._board.fill_field(row, col, val)
