import random
import math
import matplotlib.pyplot as plt
import numpy as np

coordinates = []
with open('kroA100.tsp', 'r') as file:
    content = file.readlines()
    for i in content[6:-2]:
        parts = i.split()
        coordinates.append([int(parts[0]), float(parts[1]), float(parts[2])])

# Euclidean formula
def calculate_distance(city1, city2):
    return math.sqrt((city2[1] - city1[1]) ** 2 + (city2[2] - city1[2]) ** 2)

def total_distance(route):
    distance = sum(calculate_distance(route[i], route[i+1]) for i in range(len(route) - 1))
    distance += calculate_distance(route[-1], route[0])  # Return to start
    return distance

def create_individual(coordinates):
    individual = coordinates.copy()
    random.shuffle(individual)
    return individual

def generate_population(coordinates, size):
    return [create_individual(coordinates) for _ in range(size)]

def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    return min(tournament, key=total_distance)


def ordered_crossover(parent1, parent2):
    length = len(parent1)
    start, end = sorted(random.sample(range(length), 2))
    child = [None] * length
    child[start:end+1] = parent1[start:end+1]

    filled_positions = set(range(start, end+1))
    current_index = (end + 1) % length
    for city in parent2:
        if city not in child:
            child[current_index] = city
            current_index = (current_index + 1) % length

    return child


def inversion_mutation(individual, mutation_prob):
    if random.random() < mutation_prob:
        length = len(individual)
        start, end = sorted(random.sample(range(length), 2))
        individual[start:end+1] = reversed(individual[start:end+1])
    return individual


def new_generation(previous_population, tournament_size, mutation_probability):
    new_population = []
    for _ in range(len(previous_population)):
        parent1 = tournament_selection(previous_population, tournament_size)
        parent2 = tournament_selection(previous_population, tournament_size)
        child = ordered_crossover(parent1, parent2)
        mutated_child = inversion_mutation(child, mutation_probability)
        new_population.append(mutated_child)
    return new_population

population_size = 150
generations = 1000
tournament_size = 20
mutation_probability = 0.5

population = generate_population(coordinates, population_size)

best_distances = []
for generation in range(generations):
    population = new_generation(population, tournament_size, mutation_probability)
    best_route = min(population, key=total_distance)
    best_distance = total_distance(best_route)
    best_distances.append(best_distance)

    print(f"Generation {generation + 1}: Best Distance = {best_distance}")

def info_routes(population):
    population_size = len(population)
    print(f"\nPopulation size: {population_size}")

    sorted_population = sorted(population, key=total_distance)

    best_distance = total_distance(sorted_population[0])
    print(f"Best distance in population: {best_distance}")

    median_index = population_size // 2
    median_distance = total_distance(sorted_population[median_index])
    print(f"Median distance in population: {median_distance}")

    worst_distance = total_distance(sorted_population[-1])
    print(f"Worst distance in population: {worst_distance}")
    print("")


info_routes(population)

plt.figure(figsize=(10, 6))
plt.plot(range(1, generations + 1), best_distances, marker='.', linestyle='-', color='red')
plt.title('TSP Solution Improvement over Generations')
plt.xlabel('Generations')
plt.ylabel('Best Distance')
plt.grid(True)
plt.show()

# Final result
best_route = min(population, key=total_distance)
best_distance = total_distance(best_route)
print(f"Final best solution: {best_distance}")

####################################################
#   GREEDY_ALGORITHM
####################################################
def greedy_algorithm(starting_city, cities):
    current_city = starting_city
    route = [current_city]
    cities.remove(current_city)

    while cities:
        next_city = min(cities, key=lambda city: calculate_distance(current_city, city))
        route.append(next_city)
        current_city = next_city
        cities.remove(current_city)

    route.append(starting_city)
    return route

# Greedy algorithm for every possible starting city
greedy_results = {}
for city in coordinates:
    starting_city = city
    cities_copy = coordinates.copy()
    route = greedy_algorithm(starting_city, cities_copy)
    distance = total_distance(route)
    greedy_results[starting_city[0]] = distance
    # print(f"Starting from city {starting_city[0]}: Distance = {distance}")

best_starting_city = min(greedy_results, key=greedy_results.get)
reference_score = greedy_results[best_starting_city]

random_results = []
for _ in range(100):
    random_starting_city = random.choice(coordinates)
    cities_copy = coordinates.copy()
    route = greedy_algorithm(random_starting_city, cities_copy)
    distance = total_distance(route)
    random_results.append(distance)

greedy_solutions = [greedy_algorithm(city, coordinates.copy()) for city in random.sample(coordinates, 10)]
population_size = 100
population = generate_population(coordinates, population_size)
population.extend(greedy_solutions)

print("\nResults:")
print(f"Best starting city found by greedy algorithm: {best_starting_city}, Score: {reference_score}")
print(f"Random 100 choice results: {random_results}")


####################################################
#   TEST
####################################################

# 5 runs of the greedy algorithm
greedy_algorithm_results = []
for run in range(5):
    greedy_distances = []
    for _ in range(5):
        greedy_solutions = [greedy_algorithm(city, coordinates.copy()) for city in random.sample(coordinates, 10)]
        greedy_distances.extend([total_distance(route) for route in greedy_solutions])
    greedy_algorithm_results.append(greedy_distances)

greedy_algorithm_results = np.array(greedy_algorithm_results)
greedy_algorithm_means = np.mean(greedy_algorithm_results, axis=1)
greedy_algorithm_std = np.std(greedy_algorithm_results, axis=1)
greedy_algorithm_var = np.var(greedy_algorithm_results, axis=1)

# Average statistics
print("\nStatistical Data for Greedy Algorithm over 5 runs:")
print(f"Average Mean Best Distance: {np.mean(greedy_algorithm_means)}")
print(f"Average Standard Deviation: {np.mean(greedy_algorithm_std)}")
print(f"Average Variance: {np.mean(greedy_algorithm_var)}")

# 10 runs of the genetic algorithm
genetic_algorithm_results = []
for run in range(10):
    population = generate_population(coordinates, population_size)
    best_distances = []
    for generation in range(generations):
        population = new_generation(population, tournament_size, mutation_probability)
        best_route = min(population, key=total_distance)
        best_distance = total_distance(best_route)
        best_distances.append(best_distance)
    genetic_algorithm_results.append(best_distances)

genetic_algorithm_results = np.array(genetic_algorithm_results)
genetic_algorithm_means = np.mean(genetic_algorithm_results, axis=1)
genetic_algorithm_std = np.std(genetic_algorithm_results, axis=1)
genetic_algorithm_var = np.var(genetic_algorithm_results, axis=1)

# Average statistics
print("\nStatistical Data for Genetic Algorithm over 10 runs:")
print(f"Average Mean Distance: {np.mean(genetic_algorithm_means)}")
print(f"Average Standard Deviation: {np.mean(genetic_algorithm_std)}")
print(f"Average Variance: {np.mean(genetic_algorithm_var)}")

#############################################################################
# 1000 runs of the random algorithm
random_results = []
for _ in range(1000):
    random_starting_city = random.choice(coordinates)
    cities_copy = coordinates.copy()
    route = greedy_algorithm(random_starting_city, cities_copy)
    distance = total_distance(route)
    random_results.append(distance)

best_random_value = min(random_results)
random_results = np.array(random_results)
random_mean = np.mean(random_results)
random_std = np.std(random_results)
random_var = np.var(random_results)

# Average statistics
print("\nStatistical Data for Random Algorithm over 1000 runs:")
print(f"Average Best Distance: {best_random_value}")
print(f"Average Mean Distance: {random_mean}")
print(f"Average Standard Deviation: {random_std}")
print(f"Average Variance: {random_var}")




