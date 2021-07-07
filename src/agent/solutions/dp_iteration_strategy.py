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
        change_threshold = 1e-1
        evaluation_done = False
        while not evaluation_done:
            old_state_values = self.clone_state_values()
            new_state_values = self.init_zero_state_values()
            biggest_change = 0
            for state in self.env.states.values():
                if not state.is_wall:
                    old_state_value = old_state_values[state]
                    new_state_value = self.state_value(state)
                    new_state_values[state] = new_state_value
                    biggest_change = max(biggest_change, np.abs(old_state_value - new_state_value))
            self._state_values = new_state_values
            self._deltas.append(biggest_change)
            if biggest_change < change_threshold:
                evaluation_done = True

    def policy_improvement(self):
        while True:
            policy_converged = True
            for state in self._state_values.keys():
                if state in self._policy:
                    old_action = self._policy[state]
                    new_action = None
                    best_value = float('-inf')
                    for action in self.env.actions:
                        action_value = self.action_value(state, action)
                        if action_value > best_value:
                            best_value = action_value
                            new_action = action
                    self._policy[state] = new_action
                    if new_action != old_action:
                        policy_converged = False
            if policy_converged:
                break

    def state_value(self, state):
        state_value = 0

        next_state, reward = self.env.get_new_state_and_reward(state, self.policy[state])
        next_state_val = self.discount_factor() * self._state_values[next_state]
        state_value = reward + next_state_val

        return state_value

    def action_value(self, state, action):
        action_value = 0

        next_state, action_reward = self.env.get_new_state_and_reward(state, action)
        action_value = action_reward + self.discount_factor() * self._state_values[next_state]

        return action_value
