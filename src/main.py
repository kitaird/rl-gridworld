from game2dboard import Board


def mouse_fn(btn, row, col):    # mouse calback function
    b[row][col] = 1 if not b[row][col] else 0

b = Board(6, 9)         # 3 rows, 4 columns, filled w/ None
b[0][0] = 1
b.title = "Grid world game!"
b.cell_size = 90
b.cell_color = "white"
b.on_mouse_click = mouse_fn
b.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mouse_fn()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
