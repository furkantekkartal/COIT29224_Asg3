# main.py
"""
Main script to run the Genetic Programming algorithm for symbolic regression.
This file orchestrates the entire evolutionary process.
"""

import random
import math
from problem import DATASET, FUNCTIONS
from tree import init_population
from operators import select_parents_roulette_wheel, crossover, mutation
from visualization import draw_tree

# --- GP Parameters ---
POPULATION_SIZE = 100
MAX_GENERATIONS = 100
MIN_INIT_DEPTH = 2
MAX_INIT_DEPTH = 5
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
TERMINATION_FITNESS = 0.1
RANDOM_SEED = 2 # A random seed ensures that the run is reproducible,it is for consistent results in a report.


def evaluate_individual(individual, x, y):
    """
    This function acts as the 'interpreter' for our evolved programs.
    It recursively traverses the tree to compute a final value.
    """
    # If the node is a function, recurse on its children to get arguments.
    if individual.data in [f[0] for f in FUNCTIONS.values()]:
        evaluated_children = [evaluate_individual(child, x, y) for child in individual.children]
        return individual.data(*evaluated_children)
    # If the node is a variable, substitute its value.
    elif individual.data == 'x':
        return x
    elif individual.data == 'y':
        return y
    # If the node is a constant, return its value.
    else:
        return float(individual.data)

def calculate_fitness(individual, dataset):
    """
    The 'fitness function' determines how good each solution is.
    Here, fitness is the 'sum of absolute errors' against the dataset.
    A lower score is better.
    """
    total_error = 0.0
    for x, y, expected in dataset:
        try:
            predicted = evaluate_individual(individual, x, y)
            # Penalize expressions that result in non-numeric or infinite values.
            if math.isinf(predicted) or math.isnan(predicted):
                return float('inf')
            total_error += abs(predicted - expected)
        except (ValueError, OverflowError, TypeError):
            # Handle any other evaluation errors by assigning a high penalty.
            return float('inf')
    return total_error

def evolve():
    """
    The main evolutionary loop, which follows the standard EA cycle:
    Evaluate -> Select -> Crossover -> Mutate -> Repeat.
    """
    random.seed(RANDOM_SEED)
    
    if not DATASET:
        print("Halting execution due to missing or empty dataset.")
        return

    print("--- Starting Genetic Programming for Symbolic Regression ---")
    
    # Step 1: Initialize Population
    population = init_population(POPULATION_SIZE, MIN_INIT_DEPTH, MAX_INIT_DEPTH)
    best_overall_individual = None
    best_overall_fitness = float('inf')

    # Start the generational loop.
    for gen in range(MAX_GENERATIONS):
        
        # Step 2: Evaluate the fitness of each individual in the population.
        fitnesses = [calculate_fitness(ind, DATASET) for ind in population]

        # Track the best solution found so far (a simple form of elitism for reporting).
        min_fitness_in_gen = min(fitnesses)
        if min_fitness_in_gen < best_overall_fitness:
            best_overall_fitness = min_fitness_in_gen
            best_individual_index = fitnesses.index(best_overall_fitness)
            best_overall_individual = population[best_individual_index].copy()
            print(f"Generation {gen:03d}: New Best Fitness (Error) = {best_overall_fitness:.4f}")

        # Step 3: Check termination condition.
        if best_overall_fitness < TERMINATION_FITNESS:
            print("\nTermination condition met: Found a sufficiently good solution.")
            break

        # Step 4: Create the next generation.
        new_population = []
        for _ in range(POPULATION_SIZE):
            # 4a: Selection
            parent1 = select_parents_roulette_wheel(population, fitnesses)
            
            # 4b: Crossover
            if random.random() < CROSSOVER_RATE:
                parent2 = select_parents_roulette_wheel(population, fitnesses)
                child = crossover(parent1, parent2)
            else:
                child = parent1 # Pass parent through if no crossover
            
            # 4c: Mutation
            if random.random() < MUTATION_RATE:
                mutation(child)

            new_population.append(child)
        
        # The new generation replaces the old one.
        population = new_population

    # Step 5: Report the final results.
    print("\n--- Evolution Finished ---")
    if best_overall_individual:
        print(f"Best Individual's Fitness (Final Error): {best_overall_fitness:.4f}")
        print(f"Evolved Expression (Prefix Notation): {best_overall_individual}")
        draw_tree(best_overall_individual)
    else:
        print("No valid solution was found.")

if __name__ == "__main__":
    evolve()