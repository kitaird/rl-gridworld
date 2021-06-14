from tabulate import tabulate


def pretty_print_to_console(state_values):
    print("New rewards!")
    values_to_print = [[state_values[r][c].val for c in range(len(state_values[0]))] for r in range(len(state_values))]
    print(tabulate(values_to_print, floatfmt=".3f"))
