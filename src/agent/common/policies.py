import numpy as np

from src.agent.common.value_functions import ActionValueFunction
from src.env.action import Action
from src.env.state import State
from typing import TypeAlias

# Type aliases for complex types
ActionProbabilities: TypeAlias = dict[Action, float]
StochasticPolicyMapping: TypeAlias = dict[State, ActionProbabilities]
DeterministicStateActionMapping: TypeAlias = dict[State, Action]

RNG = np.random.default_rng(seed=1)


class Policy:
    """
    Represents a policy, which defines a mapping from states to actions and their associated probabilities in a
    stochastic environment.

    The Policy class manages a stochastic policy mapping that associates states with a set of possible actions
    and their probabilities. It provides functionality for action selection based on probabilities, loading policy
    mappings from a dictionary, and accessing the probability distribution for a given state using indexing.

    Attributes:
        state_space (list[State]): The set of all possible states in the environment.
        action_space (list[Action]): The set of all possible actions in the environment.
        state_action_probabilities (StochasticPolicyMapping): A mapping of states to actions and their probabilities.
    """

    def __init__(self, state_space: list[State], action_space: list[Action],
                 state_action_probabilities: StochasticPolicyMapping):
        self._state_space = state_space
        self._action_space = action_space
        self.stochastic_policy_mapping = state_action_probabilities

    def select_action(self, state: State) -> Action:
        action_probs: ActionProbabilities = self.stochastic_policy_mapping[state]
        return RNG.choice([*action_probs.keys()], p=[*action_probs.values()])

    @staticmethod
    def load_from_dict(state_action_probabilities: StochasticPolicyMapping):
        return Policy(state_action_probabilities)  # FIXME test

    def __getitem__(self, key: State) -> ActionProbabilities:
        return self.stochastic_policy_mapping[key]


class DeterministicPolicy(Policy):
    """
    Represents a deterministic policy.

    A deterministic policy maps each state in a given state space to a single action
    in the action space. The policy ensures that for any given state, exactly one action
    is chosen with a probability of 1.0. This class is a specific implementation of
    a policy and provides methods to initialize and update the deterministic state-action
    mapping.

    Attributes:
        state_space (list[State]): List of all possible states in the environment.
        action_space (list[Action]): List of all possible actions that can be taken.
        state_action_mapping (dict[State, Action]): The deterministic mapping of states to actions.
    """

    def __init__(self, state_space: list[State], action_space: list[Action],
                 state_action_mapping: DeterministicStateActionMapping):
        probability_mapping = {state: {action: 1.0 if action == state_action_mapping[state] else 0.0
                                       for action in action_space}
                               for state in state_space
                               }
        super().__init__(state_space, action_space, probability_mapping)

    def update_action(self, state: State, action: Action) -> None:
        action_probs: ActionProbabilities = self.stochastic_policy_mapping[state]
        for a in action_probs.keys():
            action_probs[a] = 1.0 if a == action else 0.0


class RandomStochasticPolicy(Policy):
    """
    Represents a policy where actions are selected randomly with a uniform probability
    distribution over all possible actions in the given state.

    This class defines a stochastic policy, where actions are chosen at random based on a uniform probability
    distribution. It initializes the policy with a predefined state and action space, and assigns an equal probability
    to each action for a given state. This setup ensures that the policy does not favor any particular action.

    Attributes:
        state_space (list[State]): List of all possible states in the environment.
        action_space (list[Action]): List of all possible actions in the environment.
    """

    def __init__(self, state_space: list[State], action_space: list[Action]):
        uniform_probability = 1.0 / len(action_space)
        uniform_action_distribution = {state: {action: uniform_probability for action in action_space}
                                       for state in state_space
                                       }
        super().__init__(state_space, action_space, uniform_action_distribution)


class RandomDeterministicPolicy(DeterministicPolicy):
    """
    Represents a deterministic policy that assigns an action randomly chosen from the action
    space to each state in the state space.

    This class is used to define a policy where each state is deterministically mapped to an
    action, but the action is selected randomly at the time of instantiation. The random
    selection is performed for each state using the given action space.

    Attributes:
        state_space (list[State]): The list of possible states for which the policy is defined.
        action_space (list[Action]): The list of possible actions that the policy can assign to states.
    """

    def __init__(self, state_space: list[State], action_space: list[Action]):
        super().__init__(state_space, action_space, {s: RNG.choice(action_space) for s in state_space})


def get_greedy_action(action_values: ActionValueFunction, state: State) -> Action:
    """
    Selects the greedy action for a given state based on the action-value function.

    This function determines the action that has the highest estimated value for the
    provided state, according to the given action-value function. It is used in
    reinforcement learning scenarios to select the action that maximizes the expected
    reward.

    Args:
        action_values: ActionValueFunction
            The action-value function that provides a mapping from (state, action)
            pairs to their estimated values.
        state: State
            The current state for which the greedy action is to be determined.

    Returns:
        Action
            The action with the highest estimated value for the given state.
    """
    return max(action_values.action_space, key=lambda a: action_values.get(state, a))


def create_greedy_policy(action_value_function: ActionValueFunction) -> DeterministicPolicy:
    """
    Creates a deterministic policy based on the given action-value function.
    The policy selects the action with the highest value for each state with probability 1.0,
    and assigns 0.0 probability to all other actions.

    Args:
        action_value_function: Function that provides values for state-action pairs

    Returns:
        A deterministic policy that always selects the highest-value action for each state
    """
    states: list[State] = action_value_function.state_space
    actions: list[Action] = action_value_function.action_space

    policy_mapping = {state: get_greedy_action(action_value_function, state) for state in states}

    return DeterministicPolicy(states, actions, policy_mapping)


def create_epsilon_soft_policy(action_value_function: ActionValueFunction, epsilon: float) -> Policy:
    """
    Creates an epsilon-soft policy from a given action-value function. An epsilon-soft policy ensures that all
    actions have a non-zero probability of being taken, while favoring the greedy (optimal) action derived from
    the action-value function. The returned policy maps each state to a probability distribution over all
    possible actions.

    Args:
        action_value_function: An instance of ActionValueFunction that represents the state-action value
            function, which is used to calculate the greedy action and determine the probability distribution
            of actions for each state.
        epsilon: A float value representing the degree of exploration. It determines how likely the soft
            policy is to choose non-greedy actions. Must be within the range [0, 1].

    Returns:
        A Policy object that defines the epsilon-soft policy. The Policy consists of a mapping from states
        to probability distributions over actions.
    """
    state_space: list[State] = action_value_function.state_space
    action_space: list[Action] = action_value_function.action_space

    def calculate_action_probability(action: Action, greedy_a: Action) -> float:
        base_probability = epsilon / len(action_space)
        if action == greedy_a:
            return 1.0 - epsilon + base_probability
        return base_probability

    state_action_mapping = {}
    for s in state_space:
        greedy_action = get_greedy_action(action_value_function, s)
        action_probabilities = {a: calculate_action_probability(a, greedy_action) for a in action_space}
        state_action_mapping[s] = action_probabilities
    return Policy(state_space, action_space, state_action_mapping)
