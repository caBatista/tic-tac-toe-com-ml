import random

import numpy as np

from models.game_state import GameState
from models.neural_network import NeuralNetwork

class GeneticAlgorithm:
    def __init__(self, population_size=100, mutation_rate=0.01, crossover_rate=0.7):
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
        for _ in range(10):
            board = [['b'] * 3 for _ in range(3)]
            player = 'x'
            while board_checker.check_status(board) == GameState.NOT_OVER:
                if player == 'x':
                    move = individual.find_next_move(board)
                    if move:
                        board[move[0]][move[1]] = 'x'
                    else:
                        fitness -= 5
                        break
                else: 
                    difficulty = random.choice(['easy', 'medium', 'hard'])
                    move, _ = minimax.find_next_move(board, difficulty)
                    board[move[0]][move[1]] = 'o'

                player = 'o' if player == 'x' else 'x'

            final_state = board_checker.check_status(board)
            win_scores = {'easy': 5, 'medium': 10, 'hard': 15}
            draw_scores = {'easy': 2, 'medium': 5, 'hard': 8}
            loss_scores = {'easy': 15, 'medium': 10, 'hard': 5}

            if final_state == GameState.X_WON:
                fitness += win_scores[difficulty]
            elif final_state == GameState.DRAW:
                fitness += draw_scores[difficulty]
            elif final_state == GameState.O_WON:
                fitness -= loss_scores[difficulty]

        return fitness

    def select_parents(self):
        """Realiza o torneio para selecao dos pais."""
        tournament = random.sample(self.population, 3)
        best = max(tournament, key=lambda x: x.fitness)
        return best

    def crossover(self, parent1, parent2):
        """Realiza o cruzamento entre dois individuos."""
        child = NeuralNetwork()
        child.input_hidden_weights = (parent1.input_hidden_weights + parent2.input_hidden_weights) / 2
        child.hidden_output_weights = (parent1.hidden_output_weights + parent2.hidden_output_weights) / 2
        child.fitness = 0

        return child

    def mutate(self, individual):
        """Realiza a mutacao de um individuo."""
        if np.random.rand() < self.mutation_rate:
            individual.input_hidden_weights += np.random.normal(0, 0.5, individual.input_hidden_weights.shape)
            individual.hidden_output_weights += np.random.normal(0, 0.5, individual.hidden_output_weights.shape)
        return individual
    
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