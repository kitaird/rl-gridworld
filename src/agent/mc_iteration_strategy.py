import numpy as np
from src.agent.abstract_iteration_strategy import IterationStrategy


class McIterationStrategy(IterationStrategy):
    """
        This is the Monte Carlo strategy
        There is no domain knowledge available

        Implement either First-Visit or Every-Visit Monte Carlo.

        Watch out for infinite loops! Use the episode_threshold to avoid running into
        an infinite loop in the grid world! (e.g. |right|left| -> infinite loop)
    """

    def __init__(self, env):
        super().__init__("MONTE CARLO", env)
        self._returns = self.init_returns()
        self._policy = self.random_init_policy()
        self._episode_threshold = self._config.getint(self._agent_name, 'episode_threshold')

    def init_returns(self) -> {}:
        returns = {}
        for state in self.env.state_values.keys():
            returns[state] = 0 if state.is_goal else []  # We can't perform any action in the goal-state. That's why we
            # don't need an array for that state
        return returns

    def reset(self):
        super().reset()
        self._returns = self.init_returns()

    def random_init_policy(self) -> {}:
        random_init_policy = {}
        for state in self.env.states:
            random_init_policy[state.clone()] = np.random.choice(self.env.actions)
        return random_init_policy

    """
        Use the generate_trajectory method to generate states_and_returns.
        Update the values for each seen state using the mean of returns for the given state.
        Compare the difference between the current state_value and the new state_value for each state and
        log it in greatest_delta to allow plotting:
        
        self.env.deltas.append(greatest_delta)
        
        At the end of the method, append greatest_delta to self.env.deltas.  
    """
    def run_iteration_impl(self) -> None:
        greatest_delta = 0
        states_and_returns = self.generate_trajectory()
        """
            Impl here.
        """

    """
        Generate a trajectory. Use the self.env.get_random_start_state() to set the self.env.agent_state 
        for the beginning of an episode.
        
        While not reaching the goal_state, perform steps according to the agent's current policy.
        Store the next_state and reward as tuples in a list.
        WARNING: Use the episode_threshold to avoid running in an infinite loop due to the agent's policy.

        Afterwards, calculate the discounted episode_return for each visited state using the list of state-reward-tuples 
        collected earlier.
        
        This method returns a list of tuples 
            [
                (state, return), 
                (state', return')
            ]   
    """
    def generate_trajectory(self) -> []:
        """
            Impl here.
        """
        return []
