from src.alorithms.dp_iteration_strategy import DpIterationStrategy

from src.board.drl_board import DrlBoard


data = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 'g']]

cols = 9
rows = 6

board = DrlBoard(DpIterationStrategy, data)
board.show()

# move = -1 reward
# move into wall -1 (stays in same spot)
# move out of grid -1 (stays in same spot)
# move into goal 0
# move into goal -> no more jumping

# Starting with 0 reward for all fields..
