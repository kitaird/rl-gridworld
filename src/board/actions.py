from enum import Enum

from src.board.board_state import BoardCell


class Actions(Enum):
    """Possible jumps: left, right, top, bottom"""
    left = BoardCell(0, -1)
    right = BoardCell(0, 1)
    up = BoardCell(-1, 0)
    down = BoardCell(1, 0)
