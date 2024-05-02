from src.agent.abstract_iteration_strategy import IterationStrategy
from src.env.action import Action
from src.env.state import State


class McControl(IterationStrategy):
    """
        This is the Monte Carlo strategy
        There is no domain knowledge available

        Implement First-Visit and Every-Visit Monte Carlo.
    """

    def __init__(self, env):
        super().__init__("MC Control", env, use_state_values=False)
        self._action_values: dict[State, dict[Action, float]] = self.init_zero_action_values()
        self._q_returns: dict[State, dict[Action, list[float]]] = self.init_q_returns()
        self._num_of_episodes_to_collect: int = self._config.getint(self._agent_name, 'num_of_episodes_to_collect')
        self._use_every_visit: bool = self._config.getboolean(self._agent_name, 'use_every_visit')
        self._use_epsilon_greedy: bool = self._config.getboolean(self._agent_name, 'use_epsilon_greedy')
        self._epsilon: float = self._config.getfloat(self._agent_name, 'epsilon')
        self._use_true_mean: bool = self._config.getboolean(self._agent_name, 'use_true_mean')
        self._step_size: float = self._config.getfloat(self._agent_name, 'step_size')

    def init_q_returns(self) -> dict[State, dict[Action, list[float]]]:
        returns = {}
        for state in self.env.states:
            returns[state] = {action: [] for action in self.env.actions}
        return returns

    def clear(self) -> None:
        super().clear()
        self._action_values = self.init_zero_action_values()
        self._q_returns = self.init_q_returns()

    def render(self) -> None:
        super().render()
        self._plotter.pretty_print_action_values_to_console(self._action_values)

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

    def evaluate_policy(self) -> bool:
        """
            Use the generate_trajectory method to generate states_and_returns.
            Update the values for each seen state using the mean of returns for the given state.
            Compare the difference between the current state_value and the new state_value for each state and
            log it in greatest_value_delta to allow plotting:

            self.env.deltas.append(greatest_value_delta)

            At the end of the method, append greatest_value_delta to self.env.deltas.
        """
        greatest_value_delta = 0
        return True

    def compute_action_values(self, s: State, a: Action, g: float) -> float:
        """
            Compute the new action value for the given state-action pair.
            Use the formula depending on the self._use_true_mean flag: Use the true mean or a sliding window (=step size).
        """
        return 0.0

    def improve_policy(self) -> bool:
        """
            Update the policy by computing the greedy action for each state.
            If self._use_epsilon_greedy is True, use the get_epsilon_greedy_action_for_state method.
        """
        return True

    def get_epsilon_greedy_action_for_state(self, state: State) -> Action:
        """
            Implement the epsilon-greedy policy.
        """
        return Action.UP

    def get_greedy_action_for_state(self, state: State) -> Action:
        """
            Implement the greedy policy.
        """
        return Action.UP

    def generate_trajectory(self) -> [(State, Action, float)]:
        """
            Generate a trajectory.

            Follow the agent's policy until the goal (=terminated) or the timeout (=truncated) is reached.

            Afterward, calculate the discounted episode_return for each visited state using the list of
            state-reward-tuples collected earlier.

            This method returns a list of tuples
                [
                    (state_t, return_t),
                    (state_{t+1}, return_{t+1})
                ]
        """
        states_actions_returns = []
        return states_actions_returns

    @property
    def action_values(self) -> dict[State, dict[Action, float]]:
        return self._action_values
