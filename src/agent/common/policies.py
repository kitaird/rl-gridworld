import numpy as np

from src.agent.common.value_functions import ActionValueFunction
from src.env.action import Action
from src.env.state import State

RNG = np.random.default_rng(seed=1)


def random_state_action_probability_mapping(state_space: list[State], action_space: list[Action]) \
        -> dict[State, dict[Action, float]]:
    return {
        s.clone(): {a: 1.0 / len(action_space) for a in action_space}
        for s in state_space
    }

def deterministic_state_action_mapping(env, action) -> dict[State, dict[Action, float]]:
    """
        Infer the policy from scratch considering the current action_values.
    """
    state_action_mapping: dict[State, dict[Action, float]] = {}
    for state in env.valid_states:
        state_action_mapping[state] = {a: 1.0 if a == action else 0.0 for a in env.actions}
    return state_action_mapping


class Policy:

    def __init__(self, state_space: list[State], action_space: list[Action], state_action_probabilities: dict[State, dict[Action, float]] = None):
        self._state_space = state_space
        self._action_space = action_space
        self.state_action_probabilities = state_action_probabilities

    def select_action(self, state: State) -> Action:
        action_probs: dict[Action, float] = self.state_action_probabilities[state]
        return RNG.choice([*action_probs.keys()], p=[*action_probs.values()])

    @staticmethod
    def load_from_dict(state_action_probabilities: dict[State, dict[Action, float]]):
        return Policy(state_action_probabilities)

    def __getitem__(self, key: State) -> dict[Action, float]:
        return self.state_action_probabilities[key]


class DeterministicPolicy(Policy):

    def __init__(self, state_space: list[State], action_space: list[Action],
                 state_action_mapping: dict[State, dict[Action, float]] = None):
        if state_action_mapping is None:
            state_action_mapping = random_state_action_probability_mapping(state_space, action_space)
        super().__init__(state_space, action_space, state_action_mapping)

    def update_action(self, state: State, action: Action) -> None:
        action_probs: dict[Action, float] = self.state_action_probabilities[state]
        for a in action_probs.keys():
            action_probs[a] = 1.0 if a == action else 0.0


class UniformRandomPolicy(DeterministicPolicy):

    def __init__(self, state_space: list[State], action_space: list[Action]):
        state_action_probabilities = random_state_action_probability_mapping(state_space, action_space)
        super().__init__(state_space, action_space, state_action_probabilities)


def get_greedy_action(action_values: ActionValueFunction, state: State) -> Action:
    """
        Return the action with the highest value for the given state.
    """
    return max(action_values.action_space, key=lambda a: action_values.get(state, a))

def create_greedy_policy(action_value_function: ActionValueFunction) -> DeterministicPolicy:
    state_space: list[State] = action_value_function.state_space
    action_space: list[Action] = action_value_function.action_space

    state_action_mapping = {}
    for s in state_space:
        action_probabilities = {}
        optimal_a = get_greedy_action(action_value_function, s)
        for a in action_space:
            action_probabilities[a] = 1.0 if a == optimal_a else 0.0
        state_action_mapping[s] = action_probabilities
    return DeterministicPolicy(state_space, action_space, state_action_mapping)

def create_epsilon_soft_policy(action_value_function: ActionValueFunction, epsilon: float) -> Policy:
    state_space: list[State] = action_value_function.state_space
    action_space: list[Action] = action_value_function.action_space

    state_action_mapping = {}
    for s in state_space:
        action_probabilities = {}
        for a in action_space:
            if a == get_greedy_action(action_value_function, s):
                action_probabilities[a] = 1.0 - epsilon + epsilon / len(action_space)
            else:
                action_probabilities[a] = epsilon / len(action_space)
        state_action_mapping[s] = action_probabilities
    return Policy(state_space, action_space, state_action_mapping)
