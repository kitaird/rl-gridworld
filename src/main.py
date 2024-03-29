from src.agent.solutions.dp_iteration_strategy import DpIterationStrategy
from src.agent.solutions.mc_iteration_strategy import McIterationStrategy
from src.agent.solutions.td_iteration_strategy import TdIterationStrategy
from src.env.gridworld import Gridworld
from src.env.gym import Gym
from src.visualization.rl_board import RlBoard

grid_world_layout = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 1, 0, 1, 1, 0],
                     [0, 0, 1, 0, 0, 0, 1, 0, 0],
                     [0, 1, 1, 1, 1, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0, 0, 1, 0, 1],
                     [0, 1, 1, 0, 0, 1, 0, 0, 'g']]

gridworld = Gridworld(grid_world_layout)
env = Gym(gridworld)
dp_agent = DpIterationStrategy(env)
mc_agent = McIterationStrategy(env)
td_agent = TdIterationStrategy(env)
agents = [dp_agent, mc_agent, td_agent]
board = RlBoard(agents, env)
board.show()
