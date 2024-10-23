from src.agent.templates.control_algorithms.dynamic_programming.policy_iteration import PolicyIteration
from src.agent.templates.control_algorithms.dynamic_programming.value_iteration import ValueIteration
from src.agent.templates.control_algorithms.monte_carlo.on_policy_mc_control import OnPolicyMcControl
from src.agent.templates.control_algorithms.temporal_difference.q_learning import QLearning
from src.agent.templates.control_algorithms.temporal_difference.sarsa import Sarsa
from src.env.gridworld import Gridworld
from src.env.gym import Gym
from src.visualization.rl_board import RlBoard


def main():
    gridworld = Gridworld()
    env = Gym(gridworld)

    agents = [PolicyIteration(env),
              ValueIteration(env),
              OnPolicyMcControl(env),
              Sarsa(env),
              QLearning(env)]

    board = RlBoard(agents, env)
    board.show()


if __name__ == "__main__":
    main()
