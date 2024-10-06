from enum import Enum

class GameState(Enum):
    NOT_OVER = 0
    X_WON = 1
    O_WON = 2
    DRAW = 3

    def to_string(self):
        if self == GameState.X_WON:
            return "xwon"
        elif self == GameState.O_WON:
            return "owon"
        elif self == GameState.DRAW:
            return "draw"
        elif self == GameState.NOT_OVER:
            return "nver"