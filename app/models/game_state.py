from enum import Enum

class GameState(Enum):
    NOT_OVER = 0
    X_WON = 1
    O_WON = 2
    DRAW = 3
    CORRUPTED = 4

    def to_string(self):
        '''Retorna o estado em string.'''
        if self == GameState.X_WON:
            return 'X_WON'
        elif self == GameState.O_WON:
            return 'O_WON'
        elif self == GameState.DRAW:
            return 'DRAW'
        elif self == GameState.NOT_OVER:
            return 'NOT_OVER'
        elif self == GameState.CORRUPTED:
            return 'CORRUPTED'