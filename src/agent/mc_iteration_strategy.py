import numpy as np
from src.agent.abstract_iteration_strategy import IterationStrategy


class McIterationStrategy(IterationStrategy):
    """
        This is the Monte Carlo strategy
        There is no domain knowledge available

        Two options for implementation:

            * Implement an algorithm that runs episodes starting from a random position, summing up the rewards
              after each episode. Then, update the policy after each episode (not after each step!)

            * Implement an algorithm that starts always from the same position, but with epsilon-greedy to
              reach every state, summing up the rewards after each episode. Then, update the policy after each episode
              (not after each step!)

        Watch out for infinite loops! Set a timeout (e.g. 100 steps per episode) to avoid running into an infinite loop
        in the grid world! (e.g. |right|left| -> infinite loop)
    """

    def __init__(self, env):
        super().__init__(env)
        self._returns = self.init_returns()
        self._allowed_actions = self.get_allowed_actions()
        self._policy = self.random_init_policy_only_allowed_states()

    def discount_factor(self):
        return 0.75

    def init_returns(self):
        returns = {}
        for state in self._state_values.keys():
            returns[state] = 0 if state.is_goal else []
        return returns

    def run_iteration_impl(self, iterations=100):
        for _ in range(iterations):
            biggest_change = 0
            states_and_returns = self.play_game()
            seen_states = set()
            for s, G in states_and_returns:
                if s not in seen_states:
                    old_state_value = self._state_values[s]
                    self._returns[s].append(G)
                    self._state_values[s] = np.mean(self._returns[s])
                    biggest_change = max(biggest_change, np.abs(old_state_value - self._state_values[s]))
                    seen_states.add(s)
            self._deltas.append(biggest_change)
            for state in self._policy.keys():
                self._policy[state] = self.get_action_for_state(state)

    def play_game(self):
        start_fields = list(self.init_state_values().keys())
        start_index = np.random.choice(len(start_fields))
        start_state = start_fields[start_index]
        self._env.set_agent_state(start_state)

        states_and_rewards = [(self._env.agent_state(), 0)]
        iterator = 0
        while not self._env.agent_state().is_goal and iterator < 200:
            action = self._policy[self._env.agent_state()]  # TODO Problem, wenn into wall, endless loop!
            next_state, reward = self._env.get_new_state_and_reward(self._env.agent_state(), action)
            self._env.set_agent_state(next_state)
            states_and_rewards.append((next_state, reward))
            iterator += 1
        G = 0
        states_and_returns = []
        first = True
        for s, r in reversed(states_and_rewards):
            if first:
                first = False
            else:
                states_and_returns.append((s, G))
            G = r + self.discount_factor() * G
        states_and_returns.reverse()
        return states_and_returns

    def get_action_for_state(self, state):
        allowed_actions = self._allowed_actions[state]
        best_val = float('-inf')
        best_action = None
        for action in allowed_actions:
            next_state, _ = self._env.get_new_state_and_reward(state, action)
            val = self._state_values[next_state]
            if val > best_val:
                best_val = val
                best_action = action
        return best_action

    def random_init_policy_only_allowed_states(self):
        random_init_policy = {}
        for state in self._env.states.values():
            if not state.is_wall:
                possible_actions = self._allowed_actions[state]
                random_init_policy[state.clone()] = np.random.choice(possible_actions)
        return random_init_policy
