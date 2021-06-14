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

    def run_iteration_impl(self, iterations=100):
        pass

    def state_value(self, state):
        pass

    def action_value(self, state, action):
        pass
