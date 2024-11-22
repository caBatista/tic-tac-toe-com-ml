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
    avg_fitnesses = []

    pool = multiprocessing.Pool()

    for generation in range(generations):
        results = pool.map(calculate_fitness_parallel, [(individual, ga, minimax, board_checker) for individual in ga.population])
        for i, fitness in enumerate(results):
            ga.population[i].fitness = fitness

        avg_fitness = ga.find_avg_fitness()
        max_fitness = ga.find_max_fitness()
        avg_fitnesses.append(avg_fitness)
        print(f"Generation {generation}: AVG = {avg_fitness} | MAX = {max_fitness}")

        ga.population = ga.selection()
       