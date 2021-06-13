from src.alorithms.abstract_iteration_strategy import IterationStrategy


class McIterationStrategy(IterationStrategy):

    """
        This is the Monte Carlo strategy
        There is no domain knowledge available

        Implement an algorithm that runs episodes starting from a random position, summing up the rewards
        Then, update the policy after each episode (not after each step!)
    """

    def run_iteration_impl(self, iterations=100):

        pass
