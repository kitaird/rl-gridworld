from enum import Enum
from tabulate import tabulate
from src.drl_board import DrlBoard

data = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 'g']]


class Actions(Enum):
    left = (0, -1)
    right = (0, 1)
    up = (-1, 0)
    down = (1, 0)


board = DrlBoard(6, 9)


def get_init_rewards():
    return [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]


initial_rewards = get_init_rewards()


last_rewards = get_init_rewards()


def fill_field(row, col, value):
    board[row][col] = value


def init_board():
    board.title = "Grid world game!"
    board.cell_size = 90
    board.cell_color = "white"


def reward_for_cell(btn, row, col):
    print(board[row][col])


def run_iteration():
    global last_rewards
    for i in range(100):
        new_rewards = get_init_rewards()
        for row in range(0, 6):
            for col in range(0, 9):
                new_rewards[row][col] = get_reward_for_cell(row, col)
        last_rewards = new_rewards
    pretty_print_values(last_rewards)


def pretty_print_values(val):
    print("New rewards!")
    print(tabulate(val, floatfmt=".3f"))


def get_reward_from_last_rewards(new_state):
    return last_rewards[new_state[0]][new_state[1]]


def is_new_state_outside_bounds(new_state):
    return new_state[0] < 0 or new_state[0] >= len(last_rewards) or new_state[1] < 0 or new_state[1] >= len(last_rewards[0])


def is_wall(new_state):
    return data[new_state[0]][new_state[1]] == 1


def get_reward_for_cell(row, col):
    probability_for_move = 0.25

    if is_goal(row, col):
        return 0

    if is_wall((row, col)):
        return 1

    reward = 0
    reward_current_state = -1

    for action in Actions:
        new_state = (row + action.value[0], col + action.value[1])
        if is_new_state_outside_bounds(new_state) or is_wall(new_state):
            new_state = (row, col)
        reward += probability_for_move * (reward_current_state + get_reward_from_last_rewards(new_state))

    return round(reward, 3)


def is_goal(row, col):
    return data[row][col] == 'g'


init_board()
board.on_mouse_click = reward_for_cell
board.iterate_command = run_iteration
board.load(data)
board.show()

# Possible jumps: left, right, top, bottom
# move = -1 reward
# move into wall -1 (stays in same spot)
# move out of grid -1 (stays in same spot)
# move into goal 0
# move into goal -> no more jumping

# Starting with 0 reward for all fields..
