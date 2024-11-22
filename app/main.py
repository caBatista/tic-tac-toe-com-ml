from utils.training import train_network
from utils.board_checker import BoardChecker
from algorithm.minimax import Minimax
from algorithm.genetic_algorithm import GeneticAlgorithm

def main():
    board_checker = BoardChecker()
    minimax = Minimax()
    ga = GeneticAlgorithm()

    ga.initialize_population()

    print("Iniciando treinamento...")
    train_network(ga, minimax, board_checker, generations=100)

if __name__ == "__main__":
    main()
