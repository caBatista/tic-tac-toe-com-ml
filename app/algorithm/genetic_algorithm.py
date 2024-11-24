from concurrent.futures import ThreadPoolExecutor
import csv
import random
import numpy as np

from algorithm.minimax import Minimax
from utils.board_checker import BoardChecker
from models.game_state import GameState
from models.neural_network import NeuralNetwork

minimax = Minimax()
board_checker = BoardChecker()

class GeneticAlgorithm:
    NETWORK_PLAYER = 'x'
    MINIMAX_PLAYER = 'o'

    def __init__(self, population_size=200, mutation_rate=0.01, crossover_rate=1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []

    def save_population_to_csv(self, file_path):
        '''Save the GA population to a CSV file.'''
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for individual in self.population:
                writer.writerow(individual.serialize())

    def load_population_from_csv(self, file_path):
        '''Load the GA population from a CSV file.'''
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            population = []
            for row in reader:
                individual = NeuralNetwork()
                individual.fitness = 0
                individual.deserialize(row)
                population.append(individual)
            self.population = population

    def initialize_population(self):
        '''Inicializa a populacao.'''
        for _ in range(self.population_size):
            individual = NeuralNetwork()
            individual.fitness = 0
            self.population.append(individual)

    def play(self, individual, difficulty):
        '''Calcula a aptidao de um individuo.'''
        fitness = 0

        board = np.full((3, 3), 'b')
        player = self.NETWORK_PLAYER

        while board_checker.check_status(board) == GameState.NOT_OVER:
            if player == self.NETWORK_PLAYER:
                move = individual.find_next_move(board)

                if move:
                    board = self.make_move(board, move, player)

                    if difficulty in ('medium', 'hard'):
                        fitness += self.calculate_fitness_parallel(board, move)
                else:
                    return -100
            else:
                move, _ = minimax.find_next_move(board, difficulty)
                board = self.make_move(board, move, player)

            player = self.MINIMAX_PLAYER if player == self.NETWORK_PLAYER else self.NETWORK_PLAYER

        final_state = board_checker.check_status(board)
        if final_state == GameState.X_WON:
            fitness += 80
        elif final_state == GameState.DRAW:
            fitness += 40

        moves = np.count_nonzero(board == self.NETWORK_PLAYER) + np.count_nonzero(board == self.MINIMAX_PLAYER)
        fitness += 10 - moves

        return min(fitness, 100)

    def calculate_fitness_parallel(self, board, move):
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.created_two_in_line, board, move),
                executor.submit(self.created_multiple_win_paths, board, move),
                executor.submit(self.played_in_corner, move),
                executor.submit(self.blocked_oponent_win, board, move),
                executor.submit(self.created_opponent_win_paths, board, move)
            ]
            results = [future.result() for future in futures]

        fitness = 0
        if results[0]:
            fitness += 3
        if results[1]:
            fitness += 5
        if move == (1, 1):
            fitness += 2
        if results[2]:
            fitness += 2
        if results[3]:
            fitness += 8
        if results[4]:
            fitness -= 5

        return fitness

    def created_two_in_line(self, board, move):
        row, col = move
        created = False
        board[row][col] = self.NETWORK_PLAYER
        for i in range(3):
            if np.count_nonzero(board[i] == self.NETWORK_PLAYER) == 2 and np.count_nonzero(board[i] == 'b') == 1:
                created = True
                break
            if np.count_nonzero(board[:, i] == self.NETWORK_PLAYER) == 2 and np.count_nonzero(board[:, i] == 'b') == 1:
                created = True
                break
        if np.count_nonzero(np.diag(board) == self.NETWORK_PLAYER) == 2 and np.count_nonzero(np.diag(board) == 'b') == 1:
            created = True
        if np.count_nonzero(np.diag(np.fliplr(board)) == self.NETWORK_PLAYER) == 2 and np.count_nonzero(np.diag(np.fliplr(board)) == 'b') == 1:
            created = True
        board[row][col] = 'b'
        return created

    def created_multiple_win_paths(self, board, move):
        row, col = move
        board[row][col] = self.NETWORK_PLAYER
        paths = 0
        for i in range(3):
            if np.count_nonzero(board[i] == self.NETWORK_PLAYER) == 1 and np.count_nonzero(board[i] == 'b') == 2:
                paths += 1
            if np.count_nonzero(board[:, i] == self.NETWORK_PLAYER) == 1 and np.count_nonzero(board[:, i] == 'b') == 2:
                paths += 1
        if np.count_nonzero(np.diag(board) == self.NETWORK_PLAYER) == 1 and np.count_nonzero(np.diag(board) == 'b') == 2:
            paths += 1
        if np.count_nonzero(np.diag(np.fliplr(board)) == self.NETWORK_PLAYER) == 1 and np.count_nonzero(np.diag(np.fliplr(board)) == 'b') == 2:
            paths += 1
        board[row][col] = 'b'
        return paths > 1

    def played_in_corner(self, move):
        '''Verifica se a jogada foi feita em um dos cantos.'''
        return move in [(0, 0), (0, 2), (2, 0), (2, 2)]

    def blocked_oponent_win(self, board, nn_move):
        '''Identifica se a rede bloqueou a vitória do oponente.'''
        minimax_move, _ = minimax.find_next_move(board, 'hard')
        row, col = nn_move
        board[minimax_move[0]][minimax_move[1]] = self.MINIMAX_PLAYER
        blocked = False

        if board_checker.check_status(board) not in (GameState.NOT_OVER, GameState.DRAW):
            if minimax_move == nn_move:
                blocked = True

        board[row][col] = 'b'
        return blocked

    def created_opponent_win_paths(self, board, move):
        '''Verifica se a jogada criou caminhos de vitória para o oponente.'''
        row, col = move
        board[row][col] = self.NETWORK_PLAYER
        opponent_paths = 0
        for i in range(3):
            if np.count_nonzero(board[i] == self.MINIMAX_PLAYER) == 1 and np.count_nonzero(board[i] == 'b') == 2:
                opponent_paths += 1
            if np.count_nonzero(board[:, i] == self.MINIMAX_PLAYER) == 1 and np.count_nonzero(board[:, i] == 'b') == 2:
                opponent_paths += 1
        if np.count_nonzero(np.diag(board) == self.MINIMAX_PLAYER) == 1 and np.count_nonzero(np.diag(board) == 'b') == 2:
            opponent_paths += 1
        if np.count_nonzero(np.diag(np.fliplr(board)) == self.MINIMAX_PLAYER) == 1 and np.count_nonzero(np.diag(np.fliplr(board)) == 'b') == 2:
            opponent_paths += 1
        board[row][col] = 'b'
        return opponent_paths > 0

    def make_move(self, board, move, player):
        '''Realiza uma jogada.'''
        board[move[0]][move[1]] = player
        return board

    def select_parents(self):
        '''Realiza o torneio para selecao dos pais.'''
        tournament = random.sample(self.population, 3)
        best = max(tournament, key=lambda x: x.fitness)
        return best

    def crossover(self, parent1, parent2):
        '''Realiza o cruzamento entre dois individuos.'''
        child = NeuralNetwork()
        
        cut_point = np.random.randint(1, parent1.input_hidden_weights.size)
        
        child.input_hidden_weights = np.concatenate(
            (parent1.input_hidden_weights.flat[:cut_point], parent2.input_hidden_weights.flat[cut_point:])
        ).reshape(parent1.input_hidden_weights.shape)
        
        cut_point = np.random.randint(1, parent1.hidden_output_weights.size)
        
        child.hidden_output_weights = np.concatenate(
            (parent1.hidden_output_weights.flat[:cut_point], parent2.hidden_output_weights.flat[cut_point:])
        ).reshape(parent1.hidden_output_weights.shape)
        
        child.fitness = 0

        return child

    def mutate(self, individual):
        '''Realiza a mutacao de um individuo.'''
        if np.random.rand() < self.mutation_rate:
            individual.input_hidden_weights += np.random.uniform(-1, 1, individual.input_hidden_weights.shape)
            individual.hidden_output_weights += np.random.uniform(-1, 1, individual.hidden_output_weights.shape)
        return individual
    
    def adjust_mutation_rate(self, generation, max_generations):
        '''Ajusta a taxa de mutacao.'''
        self.mutation_rate = 0.01 + (0.1 - 0.01) * (1 - generation / max_generations)

    def find_best_individual(self):
        '''Encontra o melhor individuo da populacao (elitismo).'''
        return max(self.population, key=lambda x: x.fitness)

    def selection(self):
        '''Realiza a selecao dos individuos (elitismo e torneio).'''
        new_population = []

        new_population.append(self.find_best_individual())

        while len(new_population) < self.population_size:
            parent1 = self.select_parents()
            parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child))

        return new_population

    def find_avg_fitness(self):
        '''Calcula a aptidao media da populacao.'''
        return sum([individual.fitness for individual in self.population]) / self.population_size
    
    def find_max_fitness(self):
        '''Encontra a maior aptidao da populacao.'''
        return max([individual.fitness for individual in self.population])
    
    def find_min_fitness(self):
        '''Encontra a menor aptidao da populacao.'''
        return min([individual.fitness for individual in self.population])