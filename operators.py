# operators.py
"""
This module contains the core evolutionary operators used in the GP algorithm.
These operators drive the search for better solutions.
"""

import random
from tree import generate_individual

def select_parents_roulette_wheel(population, fitnesses):
    """
    Selects a parent using 'fitness proportionate (roulette wheel)' selection.
    This method gives fitter individuals a higher probability of being selected,
    but does not guarantee it, thus maintaining population diversity.
    Since lower fitness (error) is better, we invert the scores.
    """
    # Invert fitness values so that smaller errors correspond to larger slices of the wheel.
    inverted_fitnesses = [1.0 / (f + 1e-9) for f in fitnesses]
    total_inverted_fitness = sum(inverted_fitnesses)

    # Handle the edge case where all fitnesses are inf.
    if total_inverted_fitness == 0:
        return random.choice(population).copy()
        
    # Spin the wheel.
    pick = random.uniform(0, total_inverted_fitness)
    current = 0
    for i, individual in enumerate(population):
        current += inverted_fitnesses[i]
        if current > pick:
            return individual.copy()
    
    return population[-1].copy() # Fallback

def crossover(parent1, parent2):
    """
    Performs 'subtree crossover', a key genetic operator in GP.
    It works by swapping random subtrees between two parent programs,
    combining potentially useful "building blocks" from both.
    """
    offspring = parent1.copy()
    
    # Get a list of all possible crossover points in each parent.
    nodes1 = offspring.collect_nodes()
    nodes2 = parent2.collect_nodes()
    
    # Choose a random subtree from each to swap.
    crossover_point1 = random.choice(nodes1)
    crossover_point2 = random.choice(nodes2).copy()
    
    # Perform the swap.
    crossover_point1.data = crossover_point2.data
    crossover_point1.children = crossover_point2.children
    
    return offspring

def mutation(individual):
    """
    Performs 'subtree mutation', another critical genetic operator.
    It replaces a random subtree with a newly generated random one.
    This introduces new genetic material into the population, preventing
    premature convergence and exploring new parts of the solution space.
    """
    nodes = individual.collect_nodes()
    mutation_point = random.choice(nodes)
    
    # Generate a small, random tree to replace the chosen subtree.
    new_subtree = generate_individual(max_depth=3, method='grow')
    
    # Perform the replacement.
    mutation_point.data = new_subtree.data
    mutation_point.children = new_subtree.children