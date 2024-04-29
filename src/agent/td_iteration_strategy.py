import numpy as np

from src.agent.abstract_iteration_strategy import IterationStrategy
from src.env.state import State


class TdIterationStrategy(IterationStrategy):
    """
        This is the Temporal Difference learning strategy
        There is no domain knowledge available
        The state value map should be updated during episode run (online)

        Watch out for infinite loops! Use the episode_threshold to avoid running into
        an infinite loop in the grid world! (e.g. |right|left| -> infinite loop)
    """

    def __init__(self, env):
        super().__init__("TEMPORAL DIFFERENCE", env)
        self._step_size = self._config.getfloat(self._agent_name, 'step_size')
        self._episode_threshold = self._config.getint(self._agent_name, 'episode_threshold')

    def run_iteration_impl(self) -> None:
        self.policy_evaluation()
        self.policy_improvement()

    def policy_evaluation(self) -> None:
        evaluation_done = False
        while not evaluation_done:
            evaluation_done = self.evaluate_policy()

    def policy_improvement(self) -> None:
        policy_converged = False
        while not policy_converged:
            policy_converged = self.improve_policy()

    """
        Use the generate_trajectory method to generate states_and_rewards.
        Iterate through all steps of the trajectory and extract the triple (state, next_state, reward).
        Update the self._env.state_values using the newly computed TD-Error.
        Compare the difference between the current state_value and the new state_value for each state and
        log it in greatest_value_delta to allow plotting.
        At the end of the method, append greatest_value_delta to self._env.deltas:

        self._env.deltas.append(greatest_value_delta)

    """
    def evaluate_policy(self) -> bool:
        states_and_rewards = []
        greatest_value_delta = 0
        return True

    """
        Update the policy by computing the greedy action for each state using the already existing method
        self.get_greedy_action_for_state(state).
    """
    def improve_policy(self) -> bool:
        """
            Impl here.
        """
        return True

    """
        Calculate the state value.
    """

    def calculate_state_value(self, state: State, next_state: State, reward: float) -> float:
        """
            Impl here.
        """
        return 0

    """
        Calculate the TD-Error.
    """
    def td_error(self, state: State, next_state: State, reward: float) -> float:
        """
            Impl here.
        """
        return 0

    """
        Calculate the TD-Target.
    """
    def td_target(self, next_state: State, reward: float) -> float:
        """
            Impl here.
        """
        return 0

    """
        Generate a trajectory and use the self.env.get_random_start_state() to set the beginning state in 
        self.env.agent_state.

        Follow the agent's policy until the goal or the timeout (a fixed iterator of your choice, 
        self._episode_threshold) is reached.

        This method returns a list of tuples 
            [
                (state_t, reward_t), 
                (state_{t+1}, reward_{t+1})
            ]   
    """
    def generate_trajectory(self) -> [(State, float)]:
        """
            Impl here.
        """
        return []
