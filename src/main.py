from src.agent.dp_iteration_strategy import DpIterationStrategy
from src.agent.mc_iteration_strategy import McIterationStrategy
from src.agent.td_iteration_strategy import TdIterationStrategy
from src.env.environment import Environment
from src.visualization.drl_board import DrlBoard

init_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 1, 1, 1, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 'g']]

env = Environment(init_data)
dp_agent = DpIterationStrategy(env)
mc_agent = McIterationStrategy(env)
td_agent = TdIterationStrategy(env)
agents = [dp_agent, mc_agent, td_agent]
board = DrlBoard(agents, env)
board.show()

# move = -1 reward
# move into wall -1 (stays in same spot)
# move out of grid -1 (stays in same spot)
# move into goal 0
# move into goal -> no more jumping
