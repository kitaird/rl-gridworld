import numpy as np

from src.agent.abstract_iteration_strategy import IterationStrategy
from src.env.state import State


class TdPrediction(IterationStrategy):
    """
        This is the Temporal Difference learning strategy
        There is no domain knowledge available
        The state value map should be updated during episode run (online)
    """

    def __init__(self, env):
        super().__init__("TD(0) Prediction", env, use_state_values=True)
        if env.start_state is not None:
            print("-" * 100)
            print("[MC Prediction] WARNING: Start state is not None! If the policy contains loops, only a subset of states will be visited!")
            print("-" * 100)
        self._state_values: dict[State, float] = self.init_zero_state_values()
        self._step_size: float = self._config.getfloat(self._agent_name, 'step_size')

    def run_iteration_impl(self) -> None:
        self.policy_evaluation()

    def policy_evaluation(self) -> None:
        evaluation_done = False
        while not evaluation_done:
            evaluation_done = self.evaluate_policy()

    def clear(self) -> None:
        super().clear()
        self._state_values = self.init_zero_state_values()

    def render(self) -> None:
        super().render()
        self._plotter.pretty_print_state_values_to_console(self._state_values)

    def evaluate_policy(self) -> bool:
        """
            Use the generate_trajectory method to generate states_and_rewards.
            Iterate through all steps of the trajectory and extract the triple (state, next_state, reward).
            Update the self.state_values using the newly computed TD-Error.
            Compare the difference between the current state_value and the new state_value for each state and
            log it in greatest_value_delta to allow plotting.
            At the end of the method, append greatest_value_delta to self.deltas:

            self.deltas.append(greatest_value_delta)

        """
        greatest_value_delta = 0
        return True

    def calculate_state_value(self, state: State, next_state: State, reward: float) -> float:
        """
            Calculate the state value.
        """
        return 0.0

    def td_error(self, state: State, next_state: State, reward: float) -> float:
        """
            Calculate the TD-Error.
        """
        return 0.0

    def td_target(self, next_state: State, reward: float) -> float:
        """
            Calculate the TD-Target.
        """
        return 0.0

    def generate_trajectory(self) -> [(State, float)]:
        """
            Generate a trajectory .

            Follow the agent's policy until the goal (=terminated) or the timeout (=truncated) is reached.

            This method returns a list of tuples
                [
                    (state_t, reward_{t+1}),
                    (state_{t+1}, reward_{t+2})
                ]
        """
        states_rewards = []
        return states_rewards

    @property
    def state_values(self) -> dict[State, float]:
        return self._state_values
