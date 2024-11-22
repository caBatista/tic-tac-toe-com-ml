import random

import numpy as np

from models.game_state import GameState
from models.neural_network import NeuralNetwork

class GeneticAlgorithm:
    def __init__(self, population_size=200, mutation_rate=0.01, crossover_rate=1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []

    def initialize_population(self):
        """Inicializa a populacao."""
        for _ in range(self.population_size):
            individual = NeuralNetwork()
            individual.fitness = 0
            self.population.append(individual)

    def calculate_fitness(self, individual, minimax, board_checker):
        """Calcula a aptidao de um individuo."""
        fitness = 0

        board = [['b'] * 3 for _ in range(3)]
        player = 'x'

        while board_checker.check_status(board) == GameState.NOT_OVER:
            if player == 'x':
                move = individual.find_next_move(board)
                if move:
                    board[move[0]][move[1]] = 'x'
                else:
                    fitness -= 100
                    break
            else: 
                difficulty = 'medium'
                move, _ = minimax.find_next_move(board, difficulty)
                board[move[0]][move[1]] = 'o'

            player = 'o' if player == 'x' else 'x'

            final_state = board_checker.check_status(board)
            moves = board.count('x') + board.count('o')

            if final_state == GameState.O_WON:
                fitness = 0 + moves
            elif final_state == GameState.DRAW:
                fitness = 41 + moves
            elif final_state == GameState.X_WON:
                fitness = 91 + moves

        return fitness

    def select_parents(self):
        """Realiza o torneio para selecao dos pais."""
        tournament = random.sample(self.population, 3)
        best = max(tournament, key=lambda x: x.fitness)
        return best

    def crossover(self, parent1, parent2):
        """Realiza o cruzamento entre dois individuos."""
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
        """Realiza a mutacao de um individuo."""
        if np.random.rand() < self.mutation_rate:
            individual.input_hidden_weights += np.random.normal(0, 0.5, individual.input_hidden_weights.shape)
            individual.hidden_output_weights += np.random.normal(0, 0.5, individual.hidden_output_weights.shape)
        return individual
    
    def adjust_mutation_rate(self, generation, max_generations):
        """Ajusta a taxa de mutacao."""
        self.mutation_rate = 0.01 + (0.1 - 0.01) * (1 - generation / max_generations)

    def find_best_individual(self):
        """Encontra o melhor individuo da populacao (elitismo)."""
        return max(self.population, key=lambda x: x.fitness)

    def selection(self):
        """Realiza a selecao dos individuos (elitismo e torneio)."""
        new_population = []

        new_population.append(self.find_best_individual())

        while len(new_population) < self.population_size:
            parent1 = self.select_parents()
            parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child))

        return new_population

    def find_avg_fitness(self):
        """Calcula a aptidao media da populacao."""
        return sum([individual.fitness for individual in self.population]) / self.population_size
    
    def find_max_fitness(self):
        """Encontra a maior aptidao da populacao."""
        return max([individual.fitness for individual in self.population])
    