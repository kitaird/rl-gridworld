from enum import Enum

from src.env.state import State


class Action(Enum):
    """Possible jumps: left, right, top, bottom"""
    LEFT = State(0, -1)
    RIGHT = State(0, 1)
    UP = State(-1, 0)
    DOWN = State(1, 0)
