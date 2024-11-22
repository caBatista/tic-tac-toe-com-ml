import numpy as np

class RedeNeural:
    def __init__(self, entrada=9, camada_oculta=18, saida=9):
        self.entrada = entrada
        self.camada_oculta = camada_oculta
        self.saida = saida

        self.pesos_entrada_oculta = np.random.uniform(-1, 1, (entrada, camada_oculta))
        self.pesos_oculta_saida = np.random.uniform(-1, 1, (camada_oculta, saida))
        self.bias_oculta = np.random.uniform(-1, 1, camada_oculta)
        self.bias_saida = np.random.uniform(-1, 1, saida)

    def propagacao(self, entrada):
        camada_oculta = np.tanh(np.dot(entrada, self.pesos_entrada_oculta) + self.bias_oculta)
        camada_saida = np.tanh(np.dot(camada_oculta, self.pesos_oculta_saida) + self.bias_saida)
        return camada_saida

    def jogada(self, board):
        entrada = np.array([1 if cell == 'x' else -1 if cell == 'o' else 0 for row in board for cell in row])
        saida = self.propagacao(entrada)
        jogada = np.argmax(saida)
        linha, coluna = divmod(jogada, 3)
        if board[linha][coluna] == 'b':
            return linha, coluna
        else:
            return None
