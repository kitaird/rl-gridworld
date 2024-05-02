from src.agent.abstract_iteration_strategy import IterationStrategy
from src.env.state import State


class McPrediction(IterationStrategy):
    """
        This is the Monte Carlo strategy
        There is no domain knowledge available

        Implement First-Visit and Every-Visit Monte Carlo.
    """

    def __init__(self, env):
        super().__init__("MC Prediction", env, use_state_values=True)
        if env.start_state is not None:
            print("-" * 100)
            print(
                "[MC Prediction] WARNING: Start state is not None! If the policy contains loops, only a subset of states will be visited!")
            print("-" * 100)
        self._state_values: dict[State, float] = self.init_zero_state_values()
        self._returns: dict[State, list[float]] = self.init_returns()
        self._num_of_episodes_to_collect: int = self._config.getint(self._agent_name, 'num_of_episodes_to_collect')
        self._use_every_visit: bool = self._config.getboolean(self._agent_name, 'use_every_visit')

    def init_returns(self) -> dict[State, list[float]]:
        return {state: [] for state in self.env.states}

    def clear(self) -> None:
        super().clear()
        self._state_values = self.init_zero_state_values()
        self._returns = self.init_returns()

    def render(self) -> None:
        super().render()
        self._plotter.pretty_print_state_values_to_console(self._state_values)

    def run_iteration_impl(self) -> None:
        self.policy_evaluation()

    def policy_evaluation(self) -> None:
        evaluation_done = False
        while not evaluation_done:
            evaluation_done = self.evaluate_policy()

    def evaluate_policy(self) -> bool:
        """
            Use the generate_trajectory method to generate states_and_returns.
            Update the values for each seen state using the mean of returns for the given state.
            Compare the difference between the current state_value and the new state_value for each state and
            log it in greatest_value_delta to allow plotting:

            self.deltas.append(greatest_value_delta)

            At the end of the method, append greatest_value_delta to self.deltas.
        """
        greatest_value_delta = 0
        return True

    def generate_trajectory(self) -> [(State, float)]:
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
        states_and_returns = []
        return states_and_returns

    @property
    def state_values(self) -> dict[State, float]:
        return self._state_values
