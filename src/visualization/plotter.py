import matplotlib.pyplot as plt
from tabulate import tabulate
from src.env.state import State


class Plotter:

    def __init__(self, gym_environment):
        self._env = gym_environment

    def pretty_print_to_console(self):
        print("New rewards!")
        values_to_print = [[self.get_printable_state(r, c) for c in range(self._env.gridworld.cols)] for r in
                           range(self._env.gridworld.rows)]
        print(tabulate(values_to_print, floatfmt=".3f"))

    def get_printable_state(self, row, col):
        state_to_find = State(row, col)
        state_value = self._env.state_values.get(state_to_find)
        return "{:1.3f}".format(state_value) if state_value is not None else 'WALL'

    def plot_state_value_deltas(self):
        plt.plot(self._env.deltas)
        plt.xlabel("Episodes")
        plt.ylabel("State-Value Delta (highest delta per episode")
        plt.title("State-Value Convergence")
        plt.show()
