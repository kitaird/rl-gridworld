import numpy as np
from src.board.actions import Actions
from src.board.state import State


class BoardPrinter:

    def __init__(self, drl_board):
        self._board = drl_board
        self._env = drl_board.env
        self._agent = drl_board.agent

    def show_gradient(self):
        self._board.show_grid()
        state_values = self._agent.state_values()
        for row in range(len(state_values)):
            for col in range(len(state_values[0])):
                state = State(row, col)
                action_value_pairs = {}
                for action in Actions:
                    action_value_pairs[action] = self._agent.action_value(state, action)

                action_values = np.array(list(action_value_pairs.values()))
                gradients = np.sort(action_values[~np.isnan(action_values.astype(float))])
                if gradients.size != 0:
                    smallest_gradient = gradients[-1]
                    grads = {}
                    for action, action_value in action_value_pairs.items():
                        if smallest_gradient == action_value:
                            grads[action] = action_value

                    if len(grads) == 1:
                        self._board.fill_field(row, col, list(grads.keys())[0].name)

    def show_loss(self):
        self._board.show_grid()
        state_values = self._agent.state_values()
        for row in range(len(state_values)):
            for col in range(len(state_values[0])):
                state = state_values[row][col]
                if state.has_val() and not np.isnan(state.val):
                    self._board.fill_field(row, col, "{:1.3f}".format(state.val))
