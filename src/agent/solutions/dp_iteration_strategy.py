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
        knowledge.

        Implementation alternatives:
            * Policy Iteration
            * Value Iteration
    """
    def __init__(self, env):
        super().__init__("Dynamic Programming", env)
        self._policy_evaluation_threshold = 1e-2  # This threshold makes sure that the policy evaluation ends after the improvements are too insignificant
        self._last_state_values = self._env.init_zero_state_values()

    def run_iteration_impl(self) -> None:
        self.policy_evaluation()
        self.policy_improvement()

    def policy_evaluation(self) -> None:
        evaluation_done = False
        while not evaluation_done:
            evaluation_done = self.evaluate()

    def policy_improvement(self) -> None:
        while True:
            policy_converged = self.improve_policy()
            if policy_converged:
                break

    """
        This method should evaluate the current state_values and compare them to the newest state_values.
        Therefore, use the self.state_value() method to compute the new state_values for each state.
        Compare each new state_value to the current state_value.
        Store this (absolute) difference in the greatest_delta variable for each state, in order to log/plot the biggest delta
        in one evaluation iteration. Append the greatest_delta for an itaration to self._env.deltas array.
    """
    def evaluate(self) -> bool:
        greatest_delta = 0
        self._last_state_values = self.clone_state_values()
        new_state_values = self.env.init_zero_state_values()
        for state in self.env.states:
            old_state_value = self._last_state_values[state]
            new_state_value = self.state_value(state)
            new_state_values[state] = new_state_value
            greatest_delta = max(greatest_delta, np.abs(old_state_value - new_state_value))
        self.env.state_values = new_state_values
        self.env.deltas.append(greatest_delta)
        return greatest_delta <= self._policy_evaluation_threshold

    """
        This method should update self._policy for each state considering the self.action_value()
        method.
        Return a boolean that indicates if the policy has already converged (i.e. the policy already contains
        all optimal actions).
    """
    def improve_policy(self) -> bool:
        policy_converged = True
        for state in self.env.states:
            old_action = self.policy[state]
            new_action = None
            best_value = float('-inf')
            for action in self.env.actions:
                action_value = self.action_value(state, action)
                if action_value > best_value:
                    best_value = action_value
                    new_action = action
            self.policy[state] = new_action
            if new_action != old_action:
                policy_converged = False
        return policy_converged

    """
        The state_value of the provided_state (p_s) (given the agent's policy) is depending on
        the action considered most beneficial by the policy for p_s, in combination with 
        the state_value of the p_s.
        
        Use the self.env.simulate_step() to "simulate/plan" a step and see what reward
        that would return and in which next_state the agent would land.
         
        Use this next_state and reward in combination with the discounted state_value of
        p_s to get its state_value (given the agent's policy).  
    """
    def state_value(self, state) -> float:
        next_state, reward = self.env.simulate_step(state, self.policy[state])
        next_state_val = self.discount_factor * self._last_state_values[next_state]
        return reward + next_state_val

    """
        The action_value of the provided_state (p_s) and provided_action (p_a) (given the agent's policy)
        is depending on state_value of the next_state. Next_state is the state the agent
        would land in, when executing p_a in p_s. Use the discounted state_value of next_state
        in addition with the immediate reward of executing p_a in p_s to calculate the action_value.
    """
    def action_value(self, state, action) -> float:
        next_state, action_reward = self.env.simulate_step(state, action)
        next_state_val = self.discount_factor * self._last_state_values[next_state]
        return action_reward + next_state_val
