from src.agent.solutions.dp_control import DpControl
from src.agent.solutions.mc_control import McControl
from src.agent.solutions.mc_prediction import McPrediction
from src.agent.solutions.td_control import Sarsa
from src.agent.solutions.td_prediction import TdPrediction
from src.env.gridworld import Gridworld
from src.env.gym import Gym
from src.visualization.rl_board import RlBoard

grid_world_layout = [['a', 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 1, 0, 1, 1, 0],
                     [0, 0, 1, 0, 0, 0, 1, 0, 0],
                     [0, 1, 1, 1, 1, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0, 0, 1, 0, 1],
                     [0, 1, 1, 0, 0, 1, 0, 0, 'g']]


def main():
    gridworld = Gridworld(grid_world_layout)
    env = Gym(gridworld, horizon=30)

    dp_agent = DpControl(env)
    mc_prediction = McPrediction(env)
    mc_control = McControl(env)
    td_prediction = TdPrediction(env)
    sarsa = Sarsa(env)
    agents = [dp_agent, mc_prediction, mc_control, td_prediction, sarsa]

    board = RlBoard(agents, env)
    board.show()


if __name__ == "__main__":
    main()
