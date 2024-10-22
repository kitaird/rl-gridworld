import numpy as np
from tabulate import tabulate

from src.env.action import Action
from src.env.gym import Gym
from src.env.state import State


class ValueFunction:

    def __init__(self, env: Gym):
        self.env: Gym = env
        self.state_space: list[State] = env.valid_states

    def update_value(self, **kwargs) -> float:
        """
            Returns the delta between the old and the new value
        """
        raise NotImplementedError()

    def get(self, **kwargs) -> float:
        raise NotImplementedError()

    def all_values(self) -> np.ndarray:
        raise NotImplementedError()


class StateValueFunction(ValueFunction):

    def __init__(self, env: Gym, init_value: float = 0.0):
        super().__init__(env)
        self._init_value: float = init_value
        self._values: dict[State, float] = {state.clone(): self._init_value for state in self.state_space}

    def get(self, state: State) -> float:
        return self._values[state]

    def update_value(self, state: State, new_value: float) -> float:
        old_value = self._values[state]
        self._values[state] = new_value
        return np.abs(old_value - new_value)

    def all_values(self) -> np.ndarray:
        return np.array(list(self._values.values()))

    def sum(self) -> float:
        return sum(self.all_values())

    def __copy__(self):
        return self.__deepcopy__()

    def __deepcopy__(self, memodict=None):
        new = StateValueFunction(self.env, self._init_value)
        new._values = {state.clone(): value for state, value in self._values.items()}
        return new

    def __str__(self) -> str:
        values_to_print = [[self._get_printable_state_for_state_values(self.env.all_states[(row, col)])
                            for col in range(self.env.cols)] for row in range(self.env.rows)]
        return tabulate(values_to_print, floatfmt=".3f")

    def _get_printable_state_for_state_values(self, state: State) -> str:
        if state.wall:
            return '/////////'
        if state.terminal:
            return f'Terminal\n--R={state.reward}--'
        state_value = self._values[state]
        return "\n{:1.3f}\n".format(state_value)


class ActionValueFunction(ValueFunction):

    def __init__(self, env: Gym, init_value: float = 0.0):
        super().__init__(env)
        self.action_space: list[Action] = env.actions
        self._init_value: float = init_value
        self._values: dict[State, dict[Action, float]] = {state.clone(): {action: self._init_value for action in self.action_space} for state in self.state_space}

    def get(self, state: State, action: Action) -> float:
        return self._values[state][action]

    def update_value(self, state: State, action: Action, new_value: float) -> float:
        old_value = self._values[state][action]
        self._values[state][action] = new_value
        return np.abs(old_value - new_value)

    def all_values(self) -> np.ndarray:
        all_action_values = [action_value for action_values in self._values.values() for action_value in action_values.values()]
        return np.array(all_action_values)

    def sum(self) -> float:
        all_action_values = [action_value / len(action_values.keys()) for action_values in self._values.values() for action_value in action_values.values()]
        values = np.array(all_action_values)
        return sum(values)

    def __copy__(self):
        return self.__deepcopy__()

    def __deepcopy__(self, memodict=None):
        new = ActionValueFunction(self.env, self._init_value)
        new._values = {state.clone(): {action: value for action, value in action_values.items()} for
                       state, action_values in self._values.items()}
        return new

    def __str__(self) -> str:
        values_to_print = [[self._get_printable_state_for_action_values(self.env.all_states[(row, col)])
                            for col in range(self.env.cols)] for row in range(self.env.rows)]
        return tabulate(values_to_print, floatfmt=".3f")

    def _get_printable_state_for_action_values(self, state: State) -> str:
        if state.wall:
            return '----------------\n|     WALL     |\n----------------'
        if state.terminal:
            return f'----TERMINAL----\n|    R = {state.reward}    |\n----TERMINAL----'
        action_value_dict = self._values[state]
        values = action_value_dict[Action.N], action_value_dict[Action.W], action_value_dict[Action.E], action_value_dict[Action.S]
        return "|    {:1.3f}    |\n{:1.3f}  |  {:1.3f}\n|    {:1.3f}    |".format(*values)

