import csv
import multiprocessing
import os
import matplotlib.pyplot as plt

from algorithm.genetic_algorithm import GeneticAlgorithm
from algorithm.minimax import Minimax
from models.game_state import GameState
from utils.board_checker import BoardChecker

ga = GeneticAlgorithm()
minimax = Minimax()
board_checker = BoardChecker()

def visualize_evolution(avg_fitnesses):
    '''Vizualiza as metricas da evolucao'''
    plt.figure(figsize=(10, 6))
    plt.plot(avg_fitnesses, label='Média da Aptidão')
    plt.title('Evolução da Rede')
    plt.xlabel('Gerações')
    plt.ylabel('Aptidão Média')
    plt.legend()
    plt.grid()
    plt.show()

def play_parallel(args):
    '''Calcula a aptidao de um individuo paralelamente'''
    individual, difficulty = args
    return ga.play(individual=individual, difficulty=difficulty)

def define_difficulty(avg_fitness, current_difficulty):
    print(f'AVG = {avg_fitness} | DIFF = {current_difficulty.upper()}')
    if current_difficulty == 'easy' and avg_fitness >= 0:
        return 'medium'
    elif current_difficulty == 'medium' and avg_fitness >= 80:
        return 'hard'
    return current_difficulty

def train_network(generations, start_from_file=None):
    '''Treina a rede neural'''
    avg_fitnesses = []
    avg_fitness = -1000
    difficulty = 'easy'        

    if start_from_file:
        if os.path.exists(start_from_file):
            ga.load_population_from_csv(start_from_file)
            avg_fitness = ga.find_avg_fitness()
            print('População carregada. Continuando treinamento...')
        else:
            print('Nenhuma população encontrada. Criando população e iniciando treinamento...')
            ga.initialize_population()
    else:
        ga.initialize_population()
        print('População criada. Iniciando treinamento...')


    pool = multiprocessing.Pool()

    for generation in range(generations):
        difficulty = define_difficulty(avg_fitness, difficulty)

        results = pool.map(play_parallel, [(individual, difficulty) for individual in ga.population])
        for i, fitness in enumerate(results):
            ga.population[i].fitness = fitness

        avg_fitness = round(ga.find_avg_fitness(), 2)
        avg_fitnesses.append(avg_fitness)

        print(f'Generation {generation}: AVG = {avg_fitness} | INVALID = {ga.find_invalid_moves()} | DIFF = {difficulty.upper()}')

        if avg_fitness > 98:
            print(f'Stopping training as average fitness exceeded 98 in generation {generation}')
            break
        elif avg_fitness > 80:
            ga.save_population_to_csv('80_population.csv')
        elif avg_fitness > 70:
            ga.save_population_to_csv('70_population.csv')
        elif avg_fitness > 60:
            ga.save_population_to_csv('60_population.csv')
        elif avg_fitness > 0:
            ga.save_population_to_csv('basic_population.csv')
              
        ga.population = ga.selection(generation)

    best_individual = ga.find_best_individual()
    best_individual.save_to_csv('best_individual.csv')

    ga.save_population_to_csv('best_generation.csv')

    visualize_evolution(avg_fitnesses)

training_state = {
    'generation': 0,
    'board': None,
    'player': None,
    'difficulty': 'easy',
    'curr_fitness': 0,
    'avg_fitness': -1000,
    'board_state': GameState.NOT_OVER,
    'best_individual': None,
    'ga': None
}

def initialize_training():
    '''Inicializa o treinamento'''
    training_state['generation'] = 0
    training_state['curr_fitness'] = 0
    training_state['avg_fitness'] = 0
    training_state['difficulty'] = 'easy'
    training_state['player'] = None
    training_state['board'] = None
    training_state['board_state'] = GameState.NOT_OVER
    training_state['best_individual'] = None
    
    ga = GeneticAlgorithm()

    ga.initialize_population()

    training_state['ga'] = ga
    
def step_training():
    '''Simula um jogo do treinamento'''
    if training_state['ga'] is None:
        raise ValueError("Training not initialized. Call initialize_training first.")
    
    if training_state['board_state'] != GameState.NOT_OVER:
        training_state['generation'] += 1
        total_fitness = training_state['ga'].run_generation(training_state['difficulty']) + training_state['curr_fitness'] / training_state['ga'].population_size
        training_state['difficulty'] = define_difficulty(total_fitness, training_state['difficulty'])
        training_state['curr_fitness'] = 0
        training_state['avg_fitness'] = 0
        training_state['player'] = None
        training_state['board'] = None
        training_state['board_state'] = GameState.NOT_OVER
        training_state['best_individual'] = None

    difficulty = training_state['difficulty']
    board = training_state['board']
    curr_fitness = training_state['curr_fitness']
    ga = training_state['ga']
    best_individual = training_state['best_individual']

    if board is None:
        board = [['b'] * 3 for _ in range(3)]
        curr_fitness = 0
        best_individual = ga.find_best_individual()

    player = ga.NETWORK_PLAYER if training_state['player'] is None else ga.NETWORK_PLAYER if training_state['player'] == ga.MINIMAX_PLAYER else ga.MINIMAX_PLAYER

    if player == ga.NETWORK_PLAYER:
        move = best_individual.find_next_move(board)

        row, col = move

        if board[row][col] == 'b':
            board = ga.make_move(board, move, player)
            board_state = board_checker.check_status(board)
            curr_fitness = ga.calculate_fitness(board, move)
        else:
            curr_fitness -= 1000
            board_state = GameState.CORRUPTED

    else:
        move, _ = minimax.find_next_move(board, difficulty)
        board = ga.make_move(board, move, player)
        board_state = board_checker.check_status(board)

    print(board_state)

    if board_state != GameState.NOT_OVER:
        if board_state == GameState.X_WON:
            curr_fitness += 70
        elif board_state == GameState.DRAW:
            curr_fitness += 30

        moves = sum(row.count(ga.NETWORK_PLAYER) + row.count(ga.MINIMAX_PLAYER) for row in board)
        curr_fitness += 50 / moves
        avg_fitness = ga.run_generation(difficulty) + curr_fitness / ga.population_size
        training_state['avg_fitness'] = avg_fitness
        
    training_state['curr_fitness'] = curr_fitness
    training_state['board'] = board
    training_state['player'] = player
    training_state['board_state'] = board_state
    training_state['best_individual'] = best_individual

    return {
        'curr_fitness': training_state['curr_fitness'],
        'avg_fitness': training_state['avg_fitness'],
        'generation': training_state['generation'],
        'board': training_state['board'],
        'player': training_state['player'],
        'difficulty': training_state['difficulty'],
        'board_state': training_state['board_state'].to_string(),
        'last_move': (int(move[0]), int(move[1])) if move else None
    }
