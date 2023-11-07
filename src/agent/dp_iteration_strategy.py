import numpy as np
from src.agent.abstract_iteration_strategy import IterationStrategy


class DpIterationStrategy(IterationStrategy):
    """
        This is the dynamic programming strategy
        Domain knowledge is available (planning).

        With dynamic programming, there are no episodes to run, as we already have full domain
        knowledge.

        Implementation alternatives:
            * Policy Iteration
            * Value Iteration
    """

    def __init__(self, env):
        super().__init__("DYNAMIC PROGRAMMING", env)
        self._last_state_values = self._env.init_zero_state_values()
        self._policy_evaluation_threshold = self._config.getfloat(self._agent_name, 'policy_evaluation_threshold')
        # This threshold makes sure that the policy evaluation ends after the improvements are too insignificant

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
        This method should evaluate the current policy using bootstrapping.
        Compare each new state_value to the current state_value.
        Store this (absolute) difference in the greatest_delta variable for each state, in order to log/plot the biggest delta
        in one evaluation iteration. Append the greatest_delta for an itaration to self._env.deltas array:
        
        self.env.deltas.append(greatest_delta)
        
        Use self._policy_evaluation_threshold as a stopping condition when the greatest_delta becomes too small.
    """
    def evaluate_policy(self) -> bool:
        greatest_delta = 0
        """
            Impl here.
        """
        return greatest_delta <= self._policy_evaluation_threshold

    """
        This method should update self._policy for each state considering the self.action_value()
        method.
        Return a boolean that indicates if the policy has already converged (i.e. the policy already contains
        all optimal actions).
    """
    def improve_policy(self) -> bool:
        policy_converged = True
        """
            Impl here.
        """
        return policy_converged

    """
        Calculate the state_value of the given state using planning.
    """
    def state_value(self, state) -> float:
        state_value = 0
        """
            Impl here.
        """
        return state_value

    """
        Calculate the action_value of the given state-action-pair using planning.
    """
    def action_value(self, state, action) -> float:
        action_value = 0
        """
            Impl here.
        """
        return action_value
