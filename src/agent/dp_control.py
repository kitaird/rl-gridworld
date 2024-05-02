from src.agent.abstract_iteration_strategy import IterationStrategy
from src.env.state import State


class DpControl(IterationStrategy):
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
        super().__init__("DYNAMIC PROGRAMMING", env, use_state_values=True)
        self._state_values: dict[State, float] = self.init_zero_state_values()
        self._policy_evaluation_threshold: float = self._config.getfloat(self._agent_name,
                                                                         'policy_evaluation_threshold')
        # This threshold makes sure that the policy evaluation ends after the improvements are too insignificant

    def clear(self) -> None:
        super().clear()
        self._state_values = self.init_zero_state_values()

    def render(self) -> None:
        super().render()
        self._plotter.pretty_print_state_values_to_console(self._state_values)

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

    @property
    def state_values(self) -> dict[State, float]:
        return self._state_values

    def evaluate_policy(self) -> bool:
        """
            This method should evaluate the current policy using bootstrapping.
            Compare each new state_value to the current state_value.
            Store this (absolute) difference in the greatest_value_delta variable for each state, in order to log/plot the biggest delta
            in one evaluation iteration. Append the greatest_value_delta for an iteration to self.deltas array:

            self.deltas.append(greatest_value_delta)

            Use self._policy_evaluation_threshold as a stopping condition when the greatest_value_delta becomes too small.

        """
        greatest_value_delta = 0
        return greatest_value_delta <= self._policy_evaluation_threshold

    def improve_policy(self) -> bool:
        """
            This method should update self.policy for each state considering the self.action_value()
            method.
            Return a boolean that indicates if the policy has already converged (i.e. the policy already contains
            all optimal actions).

            If using value iteration, the policy should be generated from scratch using the best action for each state.
        """
        policy_converged = True
        return policy_converged

    def state_value(self, state) -> float:
        """
            Calculate the state_value of the given state using planning.
        """
        return 0.0

    def action_value(self, state, action) -> float:
        """
            Calculate the action_value of the given state-action-pair using planning.
        """
        return 0.0
