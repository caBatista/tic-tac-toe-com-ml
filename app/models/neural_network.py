import ast
import csv
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=9, hidden_layer_size=9, output_size=9):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.initialize_weights()

    def initialize_weights(self):
        '''Inicializa os pesos da populacao com valores aleatorios.'''
        self.input_hidden_weights = np.random.uniform(-1, 1, (self.input_size, self.hidden_layer_size))
        self.hidden_output_weights = np.random.uniform(-1, 1, (self.hidden_layer_size, self.output_size))
        self.hidden_bias = np.random.uniform(-1, 1, self.hidden_layer_size)
        self.output_bias = np.random.uniform(-1, 1, self.output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def propagation(self, input_data):
        hidden_layer = self.relu(np.dot(input_data, self.input_hidden_weights) + self.hidden_bias)
        output_layer = self.relu(np.dot(hidden_layer, self.hidden_output_weights) + self.output_bias)
        return output_layer
        
    def board_to_input(self, board):
        '''Converte o Board para a rede.'''
        return np.array([1 if cell == 'x' else -1 if cell == 'o' else 0 for row in board for cell in row])

    def find_next_move(self, board):
        '''Encontra a jogada a ser feita.'''
        input_data = self.board_to_input(board)
        output_data = self.propagation(input_data)
        play = np.argmax(output_data)
        x, y = divmod(play, 3)
        return (x, y)
        
    def save_to_csv(self, filename):
        '''Salva a Rede Neural em um CSV.'''
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.serialize())

    def load_from_csv(self, filename):
        '''Carrega a Rede Neural de um CSV.'''
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.deserialize(row)
        
        return self

    def serialize(self):
        '''Serializa a Rede Neural para salvar no CSV.'''
        return [
            str(self.input_hidden_weights.tolist()),
            str(self.hidden_output_weights.tolist()),
            str(self.hidden_bias.tolist()),
            str(self.output_bias.tolist())
        ]

    def deserialize(self, data):
        '''Deserializa a Rede Neural para carregar de um CSV.'''
        self.input_hidden_weights = np.array(ast.literal_eval(data[0]), dtype=float)
        self.hidden_output_weights = np.array(ast.literal_eval(data[1]), dtype=float)
        self.hidden_bias = np.array(ast.literal_eval(data[2]), dtype=float)
        self.output_bias = np.array(ast.literal_eval(data[3]), dtype=float)