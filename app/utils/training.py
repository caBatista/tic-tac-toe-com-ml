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
    individual, difficulty = args
    return ga.play(individual=individual, difficulty=difficulty)

def define_difficulty(avg_fitness, current_difficulty):
    if current_difficulty == 'easy' and avg_fitness >= 0:
        return 'medium'
    elif current_difficulty == 'medium' and avg_fitness >= 80:
        return 'hard'
    return current_difficulty

def train_network(generations, start_from_file=None):
    '''Treina a rede neural'''
    avg_fitnesses = []
    avg_fitness = -100
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

    visualize_evolution(avg_fitnesses)
    ga.save_population_to_csv('best_generation.csv')