import csv
import multiprocessing
import os
import matplotlib.pyplot as plt

from algorithm.genetic_algorithm import GeneticAlgorithm

ga = GeneticAlgorithm()

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
    individual, ga, difficulty = args
    return ga.play(individual, difficulty)

def define_difficulty(avg_fitness, current_difficulty):
    if current_difficulty == 'easy' and avg_fitness >= 0:
        return 'medium'
    elif current_difficulty == 'medium' and avg_fitness >= 80:
        return 'hard'
    return current_difficulty

def train_network(generations, start_from_scratch=True):
    '''Treina a rede neural'''
    avg_fitnesses = []
    avg_fitness = -100
    max_fitness = -100
    min_fitness = 100

    if start_from_scratch:
        ga.initialize_population()
        difficulty = 'easy'
        print('População criada. Iniciando treinamento...')
    elif not os.path.exists('medium_population.csv'):
        print('Nenhuma população encontrada. Criando população e iniciando treinamento...')
        ga.initialize_population()
        difficulty = 'easy'
    else:
        ga.load_population_from_csv('medium_population.csv')
        avg_fitness = ga.find_avg_fitness()
        difficulty = 'medium'
        print('População carregada. Continuando treinamento...')

    pool = multiprocessing.Pool()

    for generation in range(generations):
        difficulty = define_difficulty(avg_fitness, difficulty)

        results = pool.map(play_parallel, [(individual, ga, difficulty) for individual in ga.population])
        for i, fitness in enumerate(results):
            ga.population[i].fitness = fitness

        avg_fitness = ga.find_avg_fitness()
        max_fitness = ga.find_max_fitness()
        min_fitness = ga.find_min_fitness()
        avg_fitnesses.append(avg_fitness)

        print(f'Generation {generation}: AVG = {avg_fitness} | MAX = {max_fitness} | MIN = {min_fitness} | DIFF = {difficulty.upper()}')

        if avg_fitness > 98:
            print(f'Stopping training as average fitness exceeded 98 in generation {generation}')
            break
            
        ga.population = ga.selection(generation)

    visualize_evolution(avg_fitnesses)
    ga.save_population_to_csv('best_generation.csv')