from src.agent.templates.control_algorithms.dynamic_programming.policy_iteration import PolicyIteration
from src.agent.templates.control_algorithms.dynamic_programming.value_iteration import ValueIteration
from src.agent.templates.control_algorithms.learning_and_planning.dyna_q import DynaQ
from src.agent.templates.control_algorithms.monte_carlo.off_policy_mc_control import OffPolicyMcControl
from src.agent.templates.control_algorithms.monte_carlo.on_policy_mc_control import OnPolicyMcControl
from src.agent.templates.control_algorithms.n_step_bootstrapping.n_step_sarsa import NStepSarsa
from src.agent.templates.control_algorithms.n_step_bootstrapping.n_step_tree_backup import NStepTreeBackup
from src.agent.templates.control_algorithms.n_step_bootstrapping.off_policy_n_step_q_sigma import OffPolicyNStepQSigma
from src.agent.templates.control_algorithms.n_step_bootstrapping.off_policy_n_step_sarsa import OffPolicyNStepSarsa
from src.agent.templates.control_algorithms.temporal_difference.expected_sarsa import ExpectedSarsa
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
              OffPolicyMcControl(env),
              Sarsa(env),
              ExpectedSarsa(env),
              QLearning(env),
              NStepSarsa(env),
              OffPolicyNStepSarsa(env),
              NStepTreeBackup(env),
              OffPolicyNStepQSigma(env),
              DynaQ(env)]

    board = RlBoard(agents, env)
    board.show()


if __name__ == "__main__":
    main()
