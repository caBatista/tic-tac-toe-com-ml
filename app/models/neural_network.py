import json
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=9, hidden_layer_size=9, output_size=9):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.initialize_weights()

    def initialize_weights(self):
        """Inicializa os pesos da populacao com valores aleatorios."""
        self.input_hidden_weights = np.random.uniform(-1, 1, (self.input_size, self.hidden_layer_size))
        self.hidden_output_weights = np.random.uniform(-1, 1, (self.hidden_layer_size, self.output_size))
        self.hidden_bias = np.random.uniform(-1, 1, self.hidden_layer_size)
        self.output_bias = np.random.uniform(-1, 1, self.output_size)

    def sigmoid(self, x):
        """Função de ativação sigmoide."""
        return 1 / (1 + np.exp(-x))

    def propagation(self, input_data):
        """Realiza a propagacao da rede."""
        hidden_layer = self.sigmoid(np.dot(input_data, self.input_hidden_weights) + self.hidden_bias)
        output_layer = self.sigmoid(np.dot(hidden_layer, self.hidden_output_weights) + self.output_bias)
        return output_layer

    def board_to_input(self, board):
        """Converte o Board para a rede."""
        return np.array([1 if cell == 'x' else -1 if cell == 'o' else 0 for row in board for cell in row])

    def find_next_move(self, board):
        """Encontra a jogada a ser feita."""
        input_data = self.board_to_input(board)
        output_data = self.propagation(input_data)
        play = np.argmax(output_data)
        x, y = divmod(play, 3)
        return (x, y) if board[x][y] == 'b' else None
    
    def load_network_from_file(file_path):
        """Carrega a rede neural a partir de um arquivo."""
        with open(file_path, 'r') as jsonfile:
            weights = json.load(jsonfile)
        
        individual_weights = weights[0]
        
        nn = NeuralNetwork()
        
        nn.input_hidden_weights = np.array(individual_weights['input_hidden_weights'])
        nn.hidden_output_weights = np.array(individual_weights['hidden_output_weights'])
        nn.hidden_bias = np.array(individual_weights['hidden_bias'])
        nn.output_bias = np.array(individual_weights['output_bias'])
        
        return nn
    