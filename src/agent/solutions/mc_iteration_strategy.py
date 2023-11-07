import numpy as np
from src.agent.abstract_iteration_strategy import IterationStrategy


class McIterationStrategy(IterationStrategy):
    """
        This is the Monte Carlo strategy
        There is no domain knowledge available

        Implement either First-Visit or Every-Visit Monte Carlo.

        Watch out for infinite loops! Use the episode_threshold to avoid running into
        an infinite loop in the grid world! (e.g. |right|left| -> infinite loop)
    """
    def __init__(self, env):
        super().__init__("MONTE CARLO", env)
        self._returns = self.init_returns()
        self._policy = self.random_init_policy()
        self._episode_threshold = self._config.getint(self._agent_name, 'episode_threshold')

    def init_returns(self) -> {}:
        returns = {}
        for state in self.env.state_values.keys():
            returns[state] = 0 if state.is_goal else []  # We can't perform any action in the goal-state. That's why we
            # don't need an array for that state
        return returns

    def random_init_policy(self) -> {}:
        random_init_policy = {}
        for state in self.env.states:
            random_init_policy[state.clone()] = np.random.choice(self.env.actions)
        return random_init_policy

    """
        Use the generate_trajectory method to generate states_and_returns.
        Update the values for each seen state using the mean of returns for the given state.
        Compare the difference between the current state_value and the new state_value for each state and
        log it in greatest_delta to allow plotting:
        
        self.env.deltas.append(greatest_delta)
        
        At the end of the method, append greatest_delta to self.env.deltas.  
    """
    def run_iteration_impl(self) -> None:
        greatest_delta = 0
        states_and_returns = self.generate_trajectory()
        seen_states = set()
        for s, G in states_and_returns:
            if s not in seen_states:
                old_state_value = self.env.state_values[s]
                self._returns[s].append(G)
                self.env.state_values[s] = np.mean(self._returns[s])
                greatest_delta = max(greatest_delta, np.abs(old_state_value - self.env.state_values[s]))
                seen_states.add(s)
        self.env.deltas.append(greatest_delta)
        for state in self._policy.keys():
            self._policy[state] = self.get_greedy_action_for_state(state)

    """
        Generate a trajectory. Use the self.env.get_random_start_state() to set the self.env.agent_state 
        for the beginning of an episode.
        
        While not reaching the goal_state, perform steps according to the agent's current policy.
        Store the next_state and reward as tuples in a list.
        WARNING: Use the episode_threshold to avoid running in an infinite loop due to the agent's policy.

        Afterwards, calculate the discounted episode_return for each visited state using the list of state-reward-tuples 
        collected earlier.
        
        This method returns a list of tuples [(state, return), (state', return')]
    """
    def generate_trajectory(self) -> []:
        self.env.agent_state = self.env.get_random_start_state()

        states_and_rewards = [(self.env.agent_state, 0)]
        iterator = 0
        while not self.env.agent_state.is_goal and iterator < self._episode_threshold:
            action = self._policy[self.env.agent_state]
            next_state, reward = self.env.step(self.env.agent_state, action)
            states_and_rewards.append((next_state, reward))
            iterator += 1
        episode_return = 0
        states_and_returns = []
        first = True
        for state, reward in reversed(states_and_rewards):
            if first:
                first = False
            else:
                states_and_returns.append((state, episode_return))
            episode_return = reward + self.discount_factor * episode_return
        states_and_returns.reverse()
        return states_and_returns
