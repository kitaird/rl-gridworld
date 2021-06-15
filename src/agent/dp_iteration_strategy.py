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
    """

    def discount_factor(self):
        return 0.75

    def run_iteration_impl(self, iterations=5):
        for _ in range(iterations):
            self.policy_evaluation()
            self.policy_improvement()

    def policy_evaluation(self):
        state_values = self.init_state_values()
        change_threshold = 1e-4
        evaluation_done = False
        while not evaluation_done:
            biggest_change = 0
            for state in self._env.states.values():
                if not state.is_wall:
                    old_state_value = state_values[state]
                    new_state_value = self.state_value(state)
                    state_values[state] = new_state_value
                    biggest_change = max(biggest_change, np.abs(old_state_value - new_state_value))
            self._state_values = state_values
            self._deltas.append(biggest_change)
            if biggest_change > change_threshold:
                evaluation_done = True

    def policy_improvement(self):
        while True:
            policy_converged = True
            for state in self._state_values.keys():
                if state in self._policy:
                    old_action = self._policy[state]
                    new_action = None
                    best_value = float('-inf')
                    for action in self._env.actions:
                        state_value = self.action_value(state, action)
                        if state_value > best_value:
                            best_value = state_value
                            new_action = action
                    self._policy[state] = new_action
                    if new_action != old_action:
                        policy_converged = False
            if policy_converged:
                break

    def state_value(self, state):
        if state.is_goal:
            return 0

        if state.is_wall:
            raise ValueError("Error should not be wall!")

        probability_for_move = 1 / self._env.actions_dim
        cumulative_reward = 0

        for action in self._env.actions:
            next_state, reward = self._env.get_new_state_and_reward(state, action)
            next_state_val = self.discount_factor() * self.state_value_table_lookup(next_state)
            cumulative_reward += probability_for_move * (reward + next_state_val)

        return cumulative_reward

    def action_value(self, state, action):
        if state.is_goal:
            return 0

        if state.is_wall:
            raise ValueError("Error should not be wall!")

        next_state, q_reward = self._env.get_new_state_and_reward(state, action)

        reward = q_reward + self.discount_factor() * self.state_value_table_lookup(next_state)
        return reward

    def state_value_table_lookup(self, state):
        return self._state_values[state]
