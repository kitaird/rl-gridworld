from src.agent.dp_iteration_strategy import DpIterationStrategy
from src.visualization.drl_board import DrlBoard
from src.env.environment import Environment

board_layout = [['s', 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 'g']]

env = Environment(board_layout)
agent = DpIterationStrategy(env)

board = DrlBoard(agent, env)
board.show()

# move = -1 reward
# move into wall -1 (stays in same spot)
# move out of grid -1 (stays in same spot)
# move into goal 0
# move into goal -> no more jumping
