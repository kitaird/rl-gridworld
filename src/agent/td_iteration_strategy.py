import numpy as np

from src.agent.abstract_iteration_strategy import IterationStrategy


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

    """
        Use the generate_trajectory method to generate states_and_rewards.
        Iterate through all steps of the trajectory and extract the triple (state, next_state, reward).
        Update the self._env.state_values using the newly computed TD-Error.
        Compare the difference between the current state_value and the new state_value for each state and
        log it in greatest_delta to allow plotting.
        Update the policy by computing the greedy action for each state using the already existing method
        self.get_greedy_action_for_state(state).
        At the end of the method, append greatest_delta to self._env.deltas:
        
        self._env.deltas.append(greatest_delta)
        
    """
    def run_iteration_impl(self) -> None:
        states_and_rewards = self.generate_trajectory()
        greatest_delta = 0
        pass

    """
        Calculate the TD-Error.
    """
    def td_error(self, state, next_state, reward) -> float:
        """
            Impl here.
        """
        return 0

    """
        Calculate the TD-Target.
    """
    def td_target(self, next_state, reward) -> float:
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
                (state, reward), 
                (state', reward')
            ]
    """
    def generate_trajectory(self) -> []:

        """
            Impl here
        """

        return []
