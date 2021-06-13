from enum import Enum

from src.board.board_state import State


class Actions(Enum):
    """Possible jumps: left, right, top, bottom"""
    left = State(0, -1)
    right = State(0, 1)
    up = State(-1, 0)
    down = State(1, 0)
