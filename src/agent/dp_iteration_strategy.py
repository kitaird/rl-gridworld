import numpy as np
from src.agent.abstract_iteration_strategy import IterationStrategy


class DpIterationStrategy(IterationStrategy):
    """
        This is the dynamic programming strategy
        Available domain knowledge are the following:

            * Reward per step = -1
            * Reward for step into wall = -1
            * Reward for step outside boundaries = -1
            * Reward for step into goal = 0

            * A step moves the agent into a new field
            * If the new field is a wall or outside a boundary,
              the agent remains in the same field but still gets a respective reward

            * Game over when step into goal

        With dynamic programming, there are no episodes to run, as we already have full domain
        knowledge. The start position of the agent (denoted by 's' in the board_layout) may be
        ignored.

        Implementation alternatives:
            * Policy Iteration
            * Value Iteration
    """

    def discount_factor(self):
        return 0.75

    def get_iteration_size(self):
        return 5

    def get_agent_name(self):
        return "Dynamic Programming"

    def run_iteration_impl(self):
        self.policy_evaluation()
        self.policy_improvement()

    def policy_evaluation(self):
        pass

    def policy_improvement(self):
        pass

    def state_value(self, state):
        state_value = 0

        """Implement state value computation here, 
        use current policy to find action for state and resolve next state s'"""

        return state_value

    def action_value(self, state, action):
        action_value = 0

        """Implement action value computation here using self.env.get_new_state_and_reward"""

        return action_value
