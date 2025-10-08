import random

player, opponent = 'o', 'x'

class Minimax:
    def is_moves_left(self, board):
        '''Verifica se ainda existem movimentos disponiveis.'''
        for i in range(3):
            for j in range(3):
                if board[i][j] == 'b':
                    return True
        return False

    def evaluate(self, b):
        '''Avalia o tabuleiro.'''
        for row in range(3):
            if b[row][0] == b[row][1] and b[row][1] == b[row][2]:
                if b[row][0] == player:
                    return 10
                elif b[row][0] == opponent:
                    return -10

        for col in range(3):
            if b[0][col] == b[1][col] and b[1][col] == b[2][col]:
                if b[0][col] == player:
                    return 10
                elif b[0][col] == opponent:
                    return -10

        if b[0][0] == b[1][1] and b[1][1] == b[2][2]:
            if b[0][0] == player:
                return 10
            elif b[0][0] == opponent:
                return -10

        if b[0][2] == b[1][1] and b[1][1] == b[2][0]:
            if b[0][2] == player:
                return 10
            elif b[0][2] == opponent:
                return -10

        return 0

    def minimax(self, board, depth, is_max):
        '''Algoritmo Minimax.'''
        score = self.evaluate(board)

        if score == 10:
            return score

        if score == -10:
            return score

        if not self.is_moves_left(board):
            return 0

        if is_max:
            best = -1000
            for i in range(3):
                for j in range(3):
                    if board[i][j] == 'b':
                        board[i][j] = player
                        best = max(best, self.minimax(board, depth + 1, not is_max))
                        board[i][j] = 'b'
            return best
        else:
            best = 1000
            for i in range(3):
                for j in range(3):
                    if board[i][j] == 'b':
                        board[i][j] = opponent
                        best = min(best, self.minimax(board, depth + 1, not is_max))
                        board[i][j] = 'b'
            return best

    def find_next_move(self, board, difficulty):
        '''Encontra a proxima jogada.'''
        match difficulty:
            case 'hard': probability = 100
            case 'medium': probability = 50
            case 'easy': probability = 25
            case _: probability = 0

        if random.randint(0, 100) > probability:
            return self.find_random_move(board), False
        else:
            return self.find_best_move(board), True

    def find_best_move(self, board):
        '''Encontra a melhor jogada.'''
        best_val = -1000
        best_move = (-1, -1)

        for i in range(3):
            for j in range(3):
                if board[i][j] == 'b':
                    board[i][j] = player
                    move_val = self.minimax(board, 0, False)
                    board[i][j] = 'b'
                    if move_val > best_val:
                        best_move = (i, j)
                        best_val = move_val

        return best_move
    
    def find_random_move(board):
        n, m = len(board), len(board[0])
        for _ in range(n*m):              # safety cap; worst-case tries = n*m
            i, j = random.randrange(n), random.randrange(m)
            if board[i][j] == 'b':
                return (i, j)
        raise ValueError("No blank squares left")