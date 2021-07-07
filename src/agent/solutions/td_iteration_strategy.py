import numpy as np

from src.agent.abstract_iteration_strategy import IterationStrategy


class TdIterationStrategy(IterationStrategy):
    """
        This is the Temporal Difference learning strategy
        There is no domain knowledge available
        The state value map should be updated during episode run
    """

    def __init__(self, env):
        super().__init__(env)
        self._step_size = 0.1

    def get_iteration_size(self):
        return 100

    def discount_factor(self):
        return 0.75

    def get_agent_name(self):
        return "Temporal difference learning"

    def run_iteration_impl(self):
        states_and_rewards = self.generate_trajectory()
        biggest_change = 0
        for t in range(len(states_and_rewards) - 1):
            state, _ = states_and_rewards[t]
            next_state, reward = states_and_rewards[t + 1]
            old_state_value = self._state_values[state]
            new_state_value = self.new_state_value(state, next_state, reward)
            biggest_change = max(biggest_change, np.abs(old_state_value - new_state_value))
            self._state_values[state] = new_state_value
        self._deltas.append(biggest_change)
        for state in self._policy.keys():
            self._policy[state] = self.get_action_for_state(state)

    def new_state_value(self, state, next_state, reward):
        return self._state_values[state] + self._step_size * \
               (reward + self.discount_factor() * self._state_values[next_state] -
                self._state_values[state])

    def generate_trajectory(self):
        self.env.agent_state = self.env.get_random_start_state()

        states_and_rewards = [(self.env.agent_state, 0)]
        iterator = 0
        while not self.env.agent_state.is_goal and iterator < 200:
            state = self.env.agent_state
            action = self._policy[state]
            new_state, reward = self.env.get_new_state_and_reward(state, action)
            states_and_rewards.append((new_state, reward))
            self.env.agent_state = new_state
            iterator += 1
        return states_and_rewards
