import matplotlib.pyplot as plt
from tabulate import tabulate
from src.env.state import State


class Plotter:

    def __init__(self, agent):
        self._agent = agent

    def pretty_print_to_console(self):
        print("New rewards!")
        values_to_print = [[self.get_printable_state(r, c) for c in range(self._agent.env.cols)] for r in
                           range(self._agent.env.rows)]
        print(tabulate(values_to_print, floatfmt=".3f"))

    def get_printable_state(self, row, col):
        state_to_find = State(row, col)
        state_value = self._agent.state_values.get(state_to_find)
        return "{:1.3f}".format(state_value) if state_value is not None else 'WALL'

    def plot_state_value_deltas(self):
        plt.plot(self._agent.deltas)
        plt.xlabel("Episodes")
        plt.ylabel("State-Value Delta (highest delta per episode")
        plt.title("State-Value Convergence")
        plt.show()
