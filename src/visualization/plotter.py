import matplotlib.pyplot as plt
from tabulate import tabulate

from src.env.action import Action
from src.env.state import State


def plot_value_deltas(deltas):
    plt.plot(deltas)
    plt.xlabel("Iterations")
    plt.ylabel("State-Value Delta (highest delta per episode)")
    plt.title("State-Value Convergence")
    plt.show()


def get_printable_state_for_state_values(state_values, row, col):
    state_to_find = State(row, col)
    state_value = state_values.get(state_to_find)
    return "{:1.3f}".format(state_value) if state_value is not None else 'WALL'


def get_printable_state_for_action_values(action_values, row, col):
    state_to_find = State(row, col)
    if state_to_find not in action_values:
        return '----------------\n|     WALL     |\n----------------'
    action_value_dict = action_values[state_to_find]
    values = action_value_dict[Action.UP], action_value_dict[Action.LEFT], action_value_dict[Action.RIGHT], action_value_dict[Action.DOWN]
    return "|    {:1.3f}    |\n{:1.3f}  |  {:1.3f}\n|    {:1.3f}    |".format(*values)


class Plotter:

    def __init__(self, cols, rows):
        self._cols = cols
        self._rows = rows

    def pretty_print_state_values_to_console(self, state_values):
        print("New state-values!")
        values_to_print = [[get_printable_state_for_state_values(state_values, r, c) for c in range(self._cols)] for r in
                           range(self._rows)]
        print(tabulate(values_to_print, floatfmt=".3f"))

    def pretty_print_action_values_to_console(self, action_values):
        print("New action-values!")
        values_to_print = [[get_printable_state_for_action_values(action_values, r, c) for c in range(self._cols)] for r in
                           range(self._rows)]
        print(tabulate(values_to_print, floatfmt=".3f"))
