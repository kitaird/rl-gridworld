import numpy as np

from src.agent.abstract_iteration_strategy import IterationStrategy


class TdIterationStrategy(IterationStrategy):
    """
        This is the Temporal Difference learning strategy
        There is no domain knowledge available
        The state value map should be updated during episode run
    """

    def __init__(self, env):
        super().__init__("TEMPORAL DIFFERENCE", env)
        self._step_size = self._config.getfloat(self._agent_name, 'step_size')

    """
        Use the generate_trajectory method to generate states_and_rewards.
        Iterate through all steps of the trajectory and extract the triple (state, next_state, reward).
        Update the self._env.state_values using the newly computed TD-Error.
        Compare the difference between the current state_value and the new state_value for each state and
        log it in greatest_delta to allow plotting.
        Update the policy by computing the greedy action for each state using the already existing method
        self.get_greedy_action_for_state(state).
        At the end of the method, append greatest_delta to self._env.deltas.
    """
    def run_iteration_impl(self) -> None:
        states_and_rewards = self.generate_trajectory()
        greatest_delta = 0
        pass

    """
        Calculate the TD-Error using the current state_value, the step_size parameter, the current state_value as well
        as the discounted future state_value.
    """
    def td_error(self, state, next_state, reward) -> float:

        """
        Impl here

        :param state: current agent's state
        :param next_state: the state, where the agent ends up to be
        :param reward: the reward the agent got when transitioning from state to next_state
        :return: the calculated td-error considering the step-size and the current state values
        """

        return 0

    """
        Generate a trajectory and use the self.env.get_random_start_state() to set the beginning state in 
        self.env.agent_state.
        
        Follow the agent's policy until the goal or the timeout (a fixed iterator of your choice) is reached.
        
        This method returns a list of tuples [(state, reward), (state', reward')]
    """
    def generate_trajectory(self) -> []:

        """
            Impl here
        """

        return []
