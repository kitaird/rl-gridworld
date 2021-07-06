from src.agent.solutions.dp_iteration_strategy import DpIterationStrategy
from src.agent.solutions.mc_iteration_strategy import McIterationStrategy
from src.agent.solutions.td_iteration_strategy import TdIterationStrategy
from src.env.environment import Environment
from src.visualization.drl_board import DrlBoard

grid_world_layout = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 1, 0, 1, 1, 0],
                     [0, 0, 1, 0, 0, 0, 1, 0, 0],
                     [0, 1, 1, 1, 1, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0, 0, 1, 0, 1],
                     [0, 1, 1, 0, 0, 1, 0, 0, 'g']]

env = Environment(grid_world_layout)
dp_agent = DpIterationStrategy(env)
mc_agent = McIterationStrategy(env)
td_agent = TdIterationStrategy(env)
agents = [dp_agent, mc_agent, td_agent]
board = DrlBoard(agents, env)
board.show()
