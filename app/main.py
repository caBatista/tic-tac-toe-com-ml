from utils.training import train_network
from utils.board_checker import BoardChecker
from algorithm.minimax import Minimax
from algorithm.genetic_algorithm import GeneticAlgorithm

def main():
    train_network(generations=5000, start_from_scratch=True)

if __name__ == '__main__':
    main()
