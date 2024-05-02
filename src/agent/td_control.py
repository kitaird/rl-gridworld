from src.agent.abstract_iteration_strategy import IterationStrategy
from src.env.action import Action
from src.env.state import State


class Sarsa(IterationStrategy):
    """
        This is the Temporal Difference learning strategy
        There is no domain knowledge available
        The state value map should be updated during episode run (online)
    """

    def __init__(self, env):
        super().__init__("SARSA", env, use_state_values=False)
        self._action_values: dict[State, dict[Action, float]] = self.init_zero_action_values()
        self._step_size: float = self._config.getfloat(self._agent_name, 'step_size')

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

    def clear(self) -> None:
        super().clear()
        self._action_values = self.init_zero_action_values()

    def render(self) -> None:
        super().render()
        self._plotter.pretty_print_action_values_to_console(self._action_values)

    def evaluate_policy(self) -> bool:
        """
            Use the generate_trajectory method to generate states_actions_rewards.
            Iterate through all steps of the trajectory and extract the triple (state, action, reward).
            Update the self.action_values using the newly computed TD-Error.
            Compare the difference between the current action_value and the new action_value for each state-action-pair
            and log it in greatest_value_delta to allow plotting.
            At the end of the method, append greatest_value_delta to self.deltas:

            self.deltas.append(greatest_value_delta)

        """
        states_actions_rewards = self.generate_trajectory()
        return True

    def improve_policy(self) -> bool:
        """
            Update the policy by computing the greedy action for each state-action pair.
        """
        return True

    def get_greedy_action_for_state(self, state: State) -> Action:
        """
            Get the greedy action for the given state.
        """
        return Action.UP

    def calculate_action_value(self, s: State, a: Action, r: float, new_s: State, new_a: Action) -> float:
        """
            Calculate the state value.
        """
        return 0.0

    def td_error(self, s: State, a: Action, r: float, new_s: State, new_a: Action) -> float:
        """
            Calculate the TD-Error.
        """
        return 0.0

    def td_target(self, reward: float, new_s: State, new_a: Action) -> float:
        """
            Calculate the TD-Target.
        """
        return 0.0

    def generate_trajectory(self) -> [(State, float)]:
        """
            Generate a trajectory.
            Follow the agent's policy until the goal (=terminated) or the timeout (=truncated) is reached.
            This method returns a list of tuples:
                [
                    (state_t, action_t, reward_{t+1}),
                    (state_{t+1}, action_{t+1}, reward_{t+2})
                ]
        """
        states_actions_rewards = []
        return states_actions_rewards

    @property
    def action_values(self) -> dict[State, dict[Action, float]]:
        return self._action_values
