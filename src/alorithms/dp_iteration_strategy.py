from src.alorithms.abstract_iteration_strategy import IterationStrategy
from src.board.actions import Actions
from src.board.board_state import State
from src.board.board_service import get_new_state
import numpy as np


class DpIterationStrategy(IterationStrategy):
    """
        This is the dynamic programming strategy
        Available domain knowledge are the following:

            * Reward per step = -1
            * Reward for step into wall = -1
            * Reward for step outside boundaries = -1
            * Reward for step into goal = 0

            * A step moves the agent into a new field
            * If the new field is a wall or outside a boundary,
              the agent remains in the same field but still gets a respective reward

            * Game over when step into goal

        With dynamic programming, there are no episodes to run, as we already have full domain
        knowledge. The start position of the agent (denoted by 's' in the board_layout) may be
        ignored.
    """

    def run_iteration_impl(self, iterations=5):
        for i in range(iterations):
            state_values = self._board_service.init_state_values()
            for row in range(0, 6):
                for col in range(0, 9):
                    cell = State(row, col)
                    if not self._board_service.is_wall(cell):
                        state_values[row][col] = self.state_value(cell)
                    else:
                        state_values[row][col] = np.nan
            self._state_values = state_values

    def state_value(self, cell):
        if self._board_service.is_goal(cell):
            return '0'

        probability_for_move = 1 / len(Actions)
        cumulative_reward = 0

        for action in Actions:
            cumulative_reward += probability_for_move * self.action_value(cell, action)

        return "{:1.3f}".format(cumulative_reward)

    def action_value(self, state, action):
        reward_any_step = -1

        if self._board_service.is_wall(state):
            return

        next_state = get_new_state(state, action)

        if self._board_service.is_outside_bounds(next_state) or self._board_service.is_wall(next_state):
            next_state = state

        return reward_any_step + self.state_value_table_lookup(next_state)

    def state_value_table_lookup(self, cell):
        return float(self._state_values[cell.row][cell.col])
