import csv
import json
import multiprocessing
import random
import matplotlib.pyplot as plt

def visualize_evolution(avg_fitnesses):
    """Vizualiza as metricas da evolucao"""
    plt.figure(figsize=(10, 6))
    plt.plot(avg_fitnesses, label='Média da Aptidão')
    plt.title("Evolução da Rede")
    plt.xlabel("Gerações")
    plt.ylabel("Aptidão Média")
    plt.legend()
    plt.grid()
    plt.show()

def play_parallel(args):
    """Calcula a aptidao de um individuo paralelamente"""
    individual, ga, minimax, board_checker, difficulty = args
    return ga.play(individual, minimax, board_checker, difficulty)

def define_difficulty(fitness):
    if fitness > 80:
        return random.choices(["hard", "normal"], [0.7, 0.3])[0]
    elif fitness > 10:
        return random.choices(["normal", "easy"], [0.8, 0.2])[0]
    else:
        return random.choices(["easy", "normal"], [0.8, 0.2])[0]


def train_network(ga, minimax, board_checker, generations):
    """Treina a rede neural"""
    avg_fitnesses = []

    pool = multiprocessing.Pool()

    for generation in range(generations):
        ga.adjust_mutation_rate(generation, generations)
        difficulty = define_difficulty(ga.find_avg_fitness()) if avg_fitnesses else 'easy'
        results = pool.map(play_parallel, [(individual, ga, minimax, board_checker, difficulty) for individual in ga.population])
        for i, fitness in enumerate(results):
            ga.population[i].fitness = fitness

        avg_fitness = ga.find_avg_fitness()
        max_fitness = ga.find_max_fitness()
        avg_fitnesses.append(avg_fitness)

        print(f"Generation {generation}: AVG = {avg_fitness} | MAX = {max_fitness} | DIFF = {difficulty.upper()}")

        if avg_fitness > 98:
            print(f"Stopping training as average fitness exceeded 98 in generation {generation}")
            break
            
        ga.population = ga.selection()

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
        
    visualize_evolution(avg_fitnesses)