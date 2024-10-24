from models.game_state import GameState

class BoardChecker:
    def check_status(self, board):
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] != 'b':
                return GameState.X_WON if board[i][0] == 'x' else GameState.O_WON
            if board[0][i] == board[1][i] == board[2][i] != 'b':
                return GameState.X_WON if board[0][i] == 'x' else GameState.O_WON

        if board[0][0] == board[1][1] == board[2][2] != 'b':
            return GameState.X_WON if board[0][0] == 'x' else GameState.O_WON
        if board[0][2] == board[1][1] == board[2][0] != 'b':
            return GameState.X_WON if board[0][2] == 'x' else GameState.O_WON

        for row in board:
            if 'b' in row:
                return GameState.NOT_OVER
            
        return GameState.DRAW