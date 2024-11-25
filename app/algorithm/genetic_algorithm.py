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

    def __init__(self, population_size=200, mutation_rate=0.05, crossover_rate=1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        
    def save_population_to_csv(self, file_path):
        '''Salva a população em uma arquivo CSV.'''
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for individual in self.population:
                writer.writerow(individual.serialize())

    def load_population_from_csv(self, file_path):
        '''Carrega a população de um arquivo CSV.'''
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

    def play(self, individual, difficulty, games_per_generation=1):
        '''Calcula a aptidao de um individuo com base em multiplos jogos.'''
        total_fitness = 0

        for _ in range(games_per_generation):
            fitness = 0
            board = [['b'] * 3 for _ in range(3)]
            player = self.NETWORK_PLAYER

            while board_checker.check_status(board) == GameState.NOT_OVER:
                if player == self.NETWORK_PLAYER:
                    move = individual.find_next_move(board)

                    row, col = move

                    if board[row][col] == 'b':
                        board = self.make_move(board, move, player)
                        fitness += self.calculate_fitness(board, move)
                    else:
                        break
                else:
                    move, _ = minimax.find_next_move(board, difficulty)

                board = self.make_move(board, move, player)
                player = self.MINIMAX_PLAYER if player == self.NETWORK_PLAYER else self.NETWORK_PLAYER

            final_state = board_checker.check_status(board)
            if final_state == GameState.NOT_OVER:
                total_fitness -= 30
                continue
            elif final_state == GameState.X_WON:
                fitness += 70
            elif final_state == GameState.DRAW:
                fitness += 30

            moves = sum(row.count(self.NETWORK_PLAYER) + row.count(self.MINIMAX_PLAYER) for row in board)
            fitness += 50 / moves

            total_fitness += min(fitness, 100)

        return total_fitness / games_per_generation

    def run_generation(self, difficulty):
        '''Roda uma geração exceto pelo melhor individuo.'''
        total_fitness = 0

        filtered_population = list(filter(lambda x: x != self.find_best_individual(), self.population))

        for individual in filtered_population:
            total_fitness += self.play(individual, difficulty, 1)

        return total_fitness

    def make_move(self, board, move, player):
        '''Realiza a jogada.'''
        board[move[0]][move[1]] = player

        return board

    def calculate_fitness(self, board, move):
        '''Calcula a aptidao de um movimento.'''
        fitness = 1
        game_status = board_checker.check_status(board)
        
        if game_status == GameState.NOT_OVER:
            row, col = move
            player = self.NETWORK_PLAYER
            opponent = self.MINIMAX_PLAYER

            if (board[row].count(player) == 2 or 
                [board[i][col] for i in range(3)].count(player) == 2 or 
                (row == col and [board[i][i] for i in range(3)].count(player) == 2) or 
                (row + col == 2 and [board[i][2-i] for i in range(3)].count(player) == 2)):
                fitness += 30

            if (board[row].count(opponent) == 2 or 
                [board[i][col] for i in range(3)].count(opponent) == 2 or 
                (row == col and [board[i][i] for i in range(3)].count(opponent) == 2) or 
                (row + col == 2 and [board[i][2-i] for i in range(3)].count(opponent) == 2)):
                fitness += 20

            win_paths = 0
            if board[row].count(player) == 1 and board[row].count(opponent) == 0:
                win_paths += 1
            if [board[i][col] for i in range(3)].count(player) == 1 and [board[i][col] for i in range(3)].count(opponent) == 0:
                win_paths += 1
            if row == col and [board[i][i] for i in range(3)].count(player) == 1 and [board[i][i] for i in range(3)].count(opponent) == 0:
                win_paths += 1
            if row + col == 2 and [board[i][2-i] for i in range(3)].count(player) == 1 and [board[i][2-i] for i in range(3)].count(opponent) == 0:
                win_paths += 1
            if win_paths > 1:
                fitness += 10

            opponent_win_paths = 0
            if board[row].count(opponent) == 1 and board[row].count(player) == 0:
                opponent_win_paths += 1
            if [board[i][col] for i in range(3)].count(opponent) == 1 and [board[i][col] for i in range(3)].count(player) == 0:
                opponent_win_paths += 1
            if row == col and [board[i][i] for i in range(3)].count(opponent) == 1 and [board[i][i] for i in range(3)].count(player) == 0:
                opponent_win_paths += 1
            if row + col == 2 and [board[i][2-i] for i in range(3)].count(opponent) == 1 and [board[i][2-i] for i in range(3)].count(player) == 0:
                opponent_win_paths += 1
            if opponent_win_paths > 1:
                fitness += 8

            if move == (1, 1):
                fitness += 5

            if move in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                fitness += 4

            if move in [(0, 1), (1, 0), (1, 2), (2, 1)]:
                fitness += 3

        return fitness

    def selection(self, generation):
        '''Realiza a selecao dos individuos (elitismo e torneio).'''
        new_population = []

        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        new_population.extend(sorted_population[:self.population_size // 10])

        while len(new_population) < self.population_size:
            parent1 = self.select_parents()
            parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child, generation))

        new_population = self.maintain_diversity(new_population)

        return new_population

    def select_parents(self):
        '''Realiza o torneio para selecao dos pais.'''
        valid_population = [individual for individual in self.population if individual.fitness != -1000]
        sample_size = min(5, len(valid_population))

        if sample_size == 0:
            return random.choice(self.population)
        
        if sample_size != 5:
            return random.choice(valid_population)
        
        tournament = random.sample(valid_population, 5)
        
        best = max(tournament, key=lambda x: x.fitness)
        return best

    def crossover(self, parent1, parent2):
        '''Realiza o cruzamento uniforme entre dois individuos.'''
        child = NeuralNetwork()
        
        mask = np.random.rand(*parent1.input_hidden_weights.shape) > 0.5
        child.input_hidden_weights = np.where(mask, parent1.input_hidden_weights, parent2.input_hidden_weights)
        
        mask = np.random.rand(*parent1.hidden_output_weights.shape) > 0.5
        child.hidden_output_weights = np.where(mask, parent1.hidden_output_weights, parent2.hidden_output_weights)
        
        child.fitness = 0
        return child

    def mutate(self, individual, generation):
        '''Realiza a mutacao de um individuo.'''
        mutation_rate = self.mutation_rate
        if individual.fitness == -1000:
            mutation_rate *= 2

        if np.random.rand() < mutation_rate:
            mutation_strength = np.random.uniform(-0.5, 0.5, individual.input_hidden_weights.shape) * (1 / (generation + 1))
            individual.input_hidden_weights += mutation_strength
            
            mutation_strength = np.random.uniform(-0.5, 0.5, individual.hidden_output_weights.shape) * (1 / (generation + 1))
            individual.hidden_output_weights += mutation_strength
            
        return individual
    
    def maintain_diversity(self, population):
        '''Mantem a diversidade na populacao, garantindo que todos os individuos sejam unicos.'''
        unique_individuals = []
        seen = set()

        for ind in population:
            weights_tuple = (tuple(ind.input_hidden_weights.flatten()), tuple(ind.hidden_output_weights.flatten()))
            if weights_tuple not in seen:
                seen.add(weights_tuple)
                unique_individuals.append(ind)

        while len(unique_individuals) < self.population_size:
            new_individual = self.create_random_individual()
            weights_tuple = (tuple(new_individual.input_hidden_weights.flatten()), tuple(new_individual.hidden_output_weights.flatten()))
            if weights_tuple not in seen:
                seen.add(weights_tuple)
                unique_individuals.append(new_individual)

        return unique_individuals

    def create_random_individual(self):
        '''Cria um novo individuo aleatorio.'''
        new_individual = NeuralNetwork()
        new_individual.input_hidden_weights = np.random.uniform(-1, 1, new_individual.input_hidden_weights.shape)
        new_individual.hidden_output_weights = np.random.uniform(-1, 1, new_individual.hidden_output_weights.shape)
        new_individual.fitness = 0
        return new_individual

    def find_best_individual(self):
        '''Encontra o melhor individuo da populacao (elitismo).'''
        return max(self.population, key=lambda x: x.fitness)

    def find_avg_fitness(self):
        '''Calcula a aptidao media da populacao.'''
        return sum([individual.fitness for individual in self.population]) / self.population_size
    
    def find_invalid_moves(self):
        return sum([individual.fitness < 0 for individual in self.population])
    