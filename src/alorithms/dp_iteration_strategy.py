from src.alorithms.abstract_iteration_strategy import IterationStrategy
from src.board.actions import Actions
from src.board.board_printer import pretty_print_values
from src.board.board_state import BoardCell
from src.board.board_service import add_cells


class DpIterationStrategy(IterationStrategy):

    def run_iteration(self, iterations=1):
        for i in range(iterations):
            new_rewards = self._board_service.init_rewards()
            for row in range(0, 6):
                for col in range(0, 9):
                    cell = BoardCell(row, col)
                    new_rewards[row][col] = self._get_reward_for_cell(cell)
            self._last_rewards = new_rewards
        pretty_print_values(self._last_rewards)

    def _get_reward_for_cell(self, cell):
        probability_for_move = 1/len(Actions)

        if self._board_service.is_goal(cell):
            return 0

        if self._board_service.is_wall(cell):
            return -1000

        reward = 0
        reward_current_state = -1

        for action in Actions:
            next_cell = add_cells(cell, action.value)
            if self._board_service.is_outside_bounds(next_cell) or self._board_service.is_wall(next_cell):
                next_cell = cell
            reward += probability_for_move * (reward_current_state + self.get_reward_from_last_rewards(next_cell))

        return round(reward, 3)

    def get_reward_from_last_rewards(self, cell):
        return self._last_rewards[cell.row][cell.col]
