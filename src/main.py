from game2dboard import Board

from src.drl_board import DrlBoard

board = DrlBoard(6, 9)


def mouse_fn(btn, row, col):  # mouse calback function
    board[row][col] = "black" if not board[row][col] or board[row][col] == 'goal' else "goal"


def fill_field(row, col, value):
    board[row][col] = value


# def create_walls():
#     fill_field(1, 2, "black")
#     fill_field(1, 4, "black")
#     fill_field(1, 5, "black")
#     fill_field(1, 6, "black")
#     fill_field(2, 2, "black")
#     fill_field(4, 2, "black")
#     fill_field(4, 6, "black")
#     fill_field(4, 7, "black")
#     fill_field(4, 8, "black")
#
#
# def create_agent():
#     fill_field(0, 0, "agent")
#
#
# def create_goal():
#     fill_field(5, 8, "goal")


def init_board():
    board.title = "Grid world game!"
    board.cell_size = 90
    board.cell_color = "white"
    # for i in range(0, 6):
    #     for j in range(0, 9):
    #         fill_field(i, j, "white")


def reward_for_cell(btn, row, col):
    print(board[row][col])
    # if board[row][col] == 'black':
    #     print(-1)
    #     board[row][col] = -1
    # elif board[row][col] == 'agent':
    #     print(0)
    #     board[row][col] = 0
    # elif board[row][col] == 'goal':
    #     print(1)
    #     board[row][col] = 1
    # elif board[row][col] == 'white':
    #     board[row][col] = 0
    #     print(0)


init_board()
# create_walls()
# create_agent()
# create_goal()
board.on_mouse_click = reward_for_cell
data = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 'g']]
board.iterate_command = lambda: print("xxxxx")
board.load(data)
board.show()


