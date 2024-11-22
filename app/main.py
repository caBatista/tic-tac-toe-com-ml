from utils.training import train_network
from utils.board_checker import BoardChecker
from algorithm.minimax import Minimax
from models.neural_network import NeuralNetwork
from algorithm.genetic_algorithm import GeneticAlgorithm

def main():
    board_checker = BoardChecker()
    minimax = Minimax()
    ga = GeneticAlgorithm(population_size=200, mutation_rate=0.05, crossover_rate=0.8)

    ga.initialize_population()

    print("Iniciando treinamento...")
    train_network(ga, minimax, board_checker, generations=200)

if __name__ == "__main__":
    main()
