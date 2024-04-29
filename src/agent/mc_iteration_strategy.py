import numpy as np
from src.agent.abstract_iteration_strategy import IterationStrategy
from src.env.action import Action
from src.env.state import State


class McIterationStrategy(IterationStrategy):
    """
        This is the Monte Carlo strategy
        There is no domain knowledge available

        Implement First-Visit and Every-Visit Monte Carlo.

        Watch out for infinite loops! Use the episode_threshold to avoid running into
        an infinite loop in the grid world! (e.g. |right|left| -> infinite loop)
    """

    def __init__(self, env):
        super().__init__("MONTE CARLO", env)
        self._returns = self.init_returns()
        self._policy = self.random_init_policy()
        self._episode_threshold = self._config.getint(self._agent_name, 'episode_threshold')
        self._num_of_episodes_to_collect = self._config.getint(self._agent_name, 'num_of_episodes_to_collect')
        self._is_every_visit = self._config.getboolean(self._agent_name, 'is_every_visit')

    def init_returns(self) -> {State, float}:
        returns = {}
        for state in self.env.state_values.keys():
            returns[state] = 0 if state.is_goal else []  # We can't perform any action in the goal-state. That's why we
            # don't need an array for that state
        return returns

    def reset(self) -> None:
        super().reset()
        self._returns = self.init_returns()

    def random_init_policy(self) -> {State, Action}:
        random_init_policy = {}
        for state in self.env.states:
            random_init_policy[state.clone()] = self.rng.choice(self.env.actions)
        return random_init_policy

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
        Use the generate_trajectory method to generate states_and_returns.
        Update the values for each seen state using the mean of returns for the given state.
        Compare the difference between the current state_value and the new state_value for each state and
        log it in greatest_value_delta to allow plotting:
        
        self.env.deltas.append(greatest_value_delta)
        
        At the end of the method, append greatest_value_delta to self.env.deltas.  
    """
    def evaluate_policy(self) -> bool:
        greatest_value_delta = 0
        states_and_returns = []
        seen_states = set()
        """
            Impl here.
        """
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
        Generate a trajectory. Use the self.env.get_random_start_state() to set the self.env.agent_state 
        for the beginning of an episode.
        
        While not reaching the goal_state, perform steps according to the agent's current policy.
        Store the next_state and reward as tuples in a list.
        WARNING: Use the episode_threshold to avoid running in an infinite loop due to the agent's policy.

        Afterwards, calculate the discounted episode_return for each visited state using the list of state-reward-tuples 
        collected earlier.
        
        This method returns a list of tuples 
            [
                (state_t, return_t), 
                (state_{t+1}, return_{t+1})
            ]   
    """
    def generate_trajectory(self) -> [(State, float)]:
        """
            Impl here.
        """
        return []
