from enum import Enum

from src.env.state import State


class Action(Enum):
    """Possible jumps: north, south, east, west and the diagonals"""
    N = State(row=-1, col=0)
    S = State(row=1, col=0)
    E = State(row=0, col=1)
    W = State(row=0, col=-1)
    NE = State(row=-1, col=1)
    NW = State(row=-1, col=-1)
    SE = State(row=1, col=1)
    SW = State(row=1, col=-1)

cardinal_moves = [Action.N, Action.S, Action.E, Action.W]
queens_moves = [Action.N, Action.S, Action.E, Action.W, Action.NE, Action.NW, Action.SE, Action.SW]