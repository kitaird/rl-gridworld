from tabulate import tabulate


def pretty_print_values(val):
    print("New rewards!")
    print(tabulate(val, floatfmt=".3f"))
