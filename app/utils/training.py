import csv
import json
import multiprocessing
import matplotlib.pyplot as plt

def visualize_evolution(fitness_generations):
    """Vizualiza as metricas da evolucao"""
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_generations, label='Melhor Aptidão')
    plt.title("Evolução da Aptidão")
    plt.xlabel("Gerações")
    plt.ylabel("Aptidão")
    plt.legend()
    plt.grid()
    plt.show()

def calculate_fitness_parallel(args):
    """Calcula a aptidao de um individuo paralelamente"""
    individual, ga, minimax, board_checker = args
    return ga.calculate_fitness(individual, minimax, board_checker)

def train_network(ga, minimax, board_checker, generations):
    """Treina a rede neural"""
    avg_fitnesses = []

    pool = multiprocessing.Pool()

    for generation in range(generations):
        ga.adjust_mutation_rate(generation, generations)
        results = pool.map(calculate_fitness_parallel, [(individual, ga, minimax, board_checker) for individual in ga.population])
        for i, fitness in enumerate(results):
            ga.population[i].fitness = fitness

        avg_fitness = ga.find_avg_fitness()
        max_fitness = ga.find_max_fitness()
        avg_fitnesses.append(avg_fitness)
        print(f"Generation {generation}: AVG = {avg_fitness} | MAX = {max_fitness}")

        if avg_fitness > 91:
            weights = []
            for individual in ga.population:
                weights.append({
                    'input_hidden_weights': individual.input_hidden_weights.tolist(),
                    'hidden_output_weights': individual.hidden_output_weights.tolist(),
                    'hidden_bias': individual.hidden_bias.tolist(),
                    'output_bias': individual.output_bias.tolist()
                })
            with open('generation_weights.json', 'w') as jsonfile:
                json.dump(weights, jsonfile)
            print(f"Stopping training as average fitness exceeded 91 in generation {generation}")
            break

        ga.population = ga.selection()
       