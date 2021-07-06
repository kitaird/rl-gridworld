import numpy as np
from src.agent.abstract_iteration_strategy import IterationStrategy


class McIterationStrategy(IterationStrategy):
    """
        This is the Monte Carlo strategy
        There is no domain knowledge available

        Two options for implementation:

            * Implement an algorithm that runs episodes starting from a random position, summing up the rewards
              after each episode. Then, update the policy after each episode (not after each step!)

            * Implement an algorithm that starts always from the same position, but with epsilon-greedy to
              reach every state, summing up the rewards after each episode. Then, update the policy after each episode
              (not after each step!)

        Watch out for infinite loops! Set a timeout (e.g. 100 steps per episode) to avoid running into an infinite loop
        in the grid world! (e.g. |right|left| -> infinite loop)
    """

    def __init__(self, env):
        super().__init__(env)
        self._returns = self.init_returns()
        self._policy = self.random_init_policy_only_allowed_states()

    def get_iteration_size(self):
        return 100

    def discount_factor(self):
        return 0.75

    def get_agent_name(self):
        return "Monte Carlo method"

    def init_returns(self):
        pass

    def run_iteration_impl(self):
        pass

    def play_game(self):
        pass

    def random_init_policy_only_allowed_states(self):
        random_init_policy = {}
        for state in self.env.states.values():
            if not state.is_wall:
                possible_actions = self.env.allowed_actions[state]
                random_init_policy[state.clone()] = np.random.choice(possible_actions)
        return random_init_policy
