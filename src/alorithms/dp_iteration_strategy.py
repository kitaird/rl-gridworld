from src.alorithms.abstract_iteration_strategy import IterationStrategy
from src.board.actions import Actions
from src.board.visualization.console_printer import pretty_print_to_console
from src.board.board_state import BoardCell
from src.board.board_service import add_cells


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
                    cell = BoardCell(row, col)
                    state_values[row][col] = self._get_reward_for_cell(cell)
            self.V = state_values

    def _get_reward_for_cell(self, cell):
        probability_for_move = 1/len(Actions)

        if self._board_service.is_goal(cell):
            return '0'

        if self._board_service.is_wall(cell):
            return '-'

        reward = 0
        reward_current_state = -1

        for action in Actions:
            next_cell = add_cells(cell, action.value)
            if self._board_service.is_outside_bounds(next_cell) or self._board_service.is_wall(next_cell):
                next_cell = cell
            reward += probability_for_move * (reward_current_state + self.get_last_state_value(next_cell))

        return "{:1.3f}".format(reward)

    def get_last_state_value(self, cell):
        return float(self.V[cell.row][cell.col])
