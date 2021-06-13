from tabulate import tabulate


def pretty_print_to_console(val):
    print("New rewards!")
    print(tabulate(val, floatfmt=".3f"))
