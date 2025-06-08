# -*- coding: utf-8 -*-
"""
COIT29224 - Evolutionary Computation
Assignment 3: Genetic Programming for Symbolic Regression

This script implements a Genetic Programming (GP) system from scratch to solve
a symbolic regression problem. The goal is to evolve a mathematical expression
that accurately models the relationship between two independent variables (x, y)
and a dependent variable (Result) based on a given dataset.

Key features implemented as per assignment requirements:
- Representation: Expression Trees using a GPTree class.
- Initialization: Ramped half-and-half method.
- Fitness Function: Sum of absolute errors.
- Selection: Fitness proportionate (roulette wheel) selection.
- Genetic Operators: Subtree crossover and subtree mutation.
- Termination: Stops when a solution with an error < 0.1 is found or max generations are reached.
"""

import random
import operator
import copy
import math

# Try to import graphviz, but handle the case where it's not installed.
try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Warning: graphviz library not found. Tree visualization will be skipped.")
    print("To install: pip install graphviz")

# -----------------------------------------------------------------------------
# 1. GP PARAMETERS
# -----------------------------------------------------------------------------
POPULATION_SIZE = 100       # Number of individuals in the population.
MAX_GENERATIONS = 100       # Maximum number of generations to run.
MIN_INIT_DEPTH = 2          # Minimum depth for initial trees.
MAX_INIT_DEPTH = 5          # Maximum depth for initial trees.
CROSSOVER_RATE = 0.8        # Probability of performing crossover.
MUTATION_RATE = 0.2         # Probability of performing mutation for an individual.
TERMINATION_FITNESS = 0.1   # Fitness threshold to stop evolution.

# -----------------------------------------------------------------------------
# 2. PROBLEM DEFINITION (Functions, Terminals, and Dataset)
# -----------------------------------------------------------------------------

def safe_div(x, y):
    """
    Protected division to avoid division-by-zero errors.
    Returns 1 if the denominator is close to zero.
    """
    return x / y if abs(y) > 1e-6 else 1.0

# Function Set: The operations the GP can use.
FUNCTIONS = {
    'add': (operator.add, 2),
    'sub': (operator.sub, 2),
    'mul': (operator.mul, 2),
    'div': (safe_div, 2)
}

# Terminal Set: The inputs and constants the GP can use.
TERMINALS = ['x', 'y']
CONSTANT_RANGE = (-5.0, 5.0) # Range for generating random constants.

# Dataset from Assignment "Table 1"
DATASET = [
    (-1, -1, -6.33333), (-1, 0, -6.0), (-1, 1, -5.66667), (-1, 2, -5.33333),
    (-1, 3, -5.0), (-1, 4, -4.66667), (-1, 5, -4.33333),
    (0, -1, -4.33333), (0, 0, -4.0), (0, 1, -3.66667), (0, 2, -3.33333),
    (0, 3, -3.0), (0, 4, -2.66667), (0, 5, -2.33333),
    (1, -1, -2.33333), (1, 0, -2.0), (1, 1, -1.66667), (1, 2, -1.33333),
    (1, 3, -1.0), (1, 4, -0.666667), (1, 5, -0.333333),
    (2, -1, -0.333333), (2, 0, 0.0), (2, 1, 0.333333), (2, 2, 0.666667),
    (2, 3, 1.0), (2, 4, 1.33333), (2, 5, 1.66667),
    (3, -1, 1.66667), (3, 0, 2.0), (3, 1, 2.33333), (3, 2, 2.66667),
    (3, 3, 3.0), (3, 4, 3.33333), (3, 5, 3.66667),
    (4, -1, 3.66667), (4, 0, 4.0), (4, 1, 4.33333), (4, 2, 4.66667),
    (4, 3, 5.0), (4, 4, 5.33333), (4, 5, 5.66667),
    (5, -1, 5.66667), (5, 0, 6.0), (5, 1, 6.33333), (5, 2, 6.66667),
    (5, 3, 7.0), (5, 4, 7.33333), (5, 5, 7.66667)
]

# -----------------------------------------------------------------------------
# 3. GP CORE IMPLEMENTATION (Tree, Fitness, Operators)
# -----------------------------------------------------------------------------

class GPTree:
    """
    Represents an individual program as an expression tree.
    Each node contains data (a function or a terminal) and has children.
    """
    def __init__(self, data=None, children=None):
        self.data = data
        self.children = children if children else []

    def __str__(self):
        """Returns the expression in prefix notation (e.g., "+ x 1")."""
        if not self.children:
            return str(self.data)
        return f"({self.get_node_name()} {' '.join(str(c) for c in self.children)})"

    def get_node_name(self):
        """Returns a string label for the node's data."""
        # Check if the data is a function from the FUNCTIONS dict
        for name, (func, arity) in FUNCTIONS.items():
            if func == self.data:
                return name
        # Otherwise, it's a terminal
        return str(self.data)

    def size(self):
        """Recursively count the total number of nodes in the tree."""
        return 1 + sum(child.size() for child in self.children)

    def collect_nodes(self):
        """Return a flat list of all nodes in this subtree."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.collect_nodes())
        return nodes

    def copy(self):
        """Create a deep copy of this tree."""
        return copy.deepcopy(self)


def generate_individual(max_depth, method='grow', current_depth=0):
    """
    Generates a single random expression tree (individual).
    This function implements the core logic for the 'grow' and 'full' methods.

    Args:
        max_depth (int): The maximum depth of the tree to be generated.
        method (str): The generation method, either 'grow' or 'full'.
        current_depth (int): The current depth during recursion.

    Returns:
        GPTree: A randomly generated expression tree.
    """
    # Base case for recursion: if max depth is reached, force a terminal.
    if current_depth >= max_depth or (method == 'grow' and random.random() < 0.5):
        # Pick a terminal: either a variable 'x'/'y' or a random constant
        if random.random() < 0.5:
             terminal = random.choice(TERMINALS)
        else:
             terminal = round(random.uniform(*CONSTANT_RANGE), 2)
        return GPTree(data=terminal)

    # Recursive step: pick a function and generate its children.
    func_name = random.choice(list(FUNCTIONS.keys()))
    func, arity = FUNCTIONS[func_name]
    
    children = []
    for _ in range(arity):
        child = generate_individual(max_depth, method, current_depth + 1)
        children.append(child)

    return GPTree(data=func, children=children)


def init_population():
    """
    Creates the initial population using the Ramped Half-and-Half method.
    This ensures a diverse set of trees with varying depths and shapes.
    """
    population = []
    for depth in range(MIN_INIT_DEPTH, MAX_INIT_DEPTH + 1):
        # Half 'full'
        for _ in range(POPULATION_SIZE // (2 * (MAX_INIT_DEPTH - MIN_INIT_DEPTH + 1))):
            population.append(generate_individual(depth, method='full'))
        # Half 'grow'
        for _ in range(POPULATION_SIZE // (2 * (MAX_INIT_DEPTH - MIN_INIT_DEPTH + 1))):
            population.append(generate_individual(depth, method='grow'))

    # Fill up to POPULATION_SIZE if integer division left some spots
    while len(population) < POPULATION_SIZE:
        depth = random.randint(MIN_INIT_DEPTH, MAX_INIT_DEPTH)
        population.append(generate_individual(depth, method='grow'))
        
    return population


def evaluate_individual(individual, x, y):
    """
    Recursively evaluates the expression tree for a given x and y.

    Args:
        individual (GPTree): The expression tree to evaluate.
        x (float): The value for the 'x' terminal.
        y (float): The value for the 'y' terminal.

    Returns:
        float: The result of the expression.
    """
    # If it's a function node, evaluate children and apply the function.
    if individual.data in [f[0] for f in FUNCTIONS.values()]:
        evaluated_children = [evaluate_individual(child, x, y) for child in individual.children]
        return individual.data(*evaluated_children)
    
    # If it's a terminal node, return its value.
    elif individual.data == 'x':
        return x
    elif individual.data == 'y':
        return y
    else: # It's a constant
        return float(individual.data)


def calculate_fitness(individual, dataset):
    """
    Calculates the fitness of an individual as the sum of absolute errors
    over the entire dataset. Lower is better.
    """
    total_error = 0.0
    for x, y, expected_result in dataset:
        try:
            predicted_result = evaluate_individual(individual, x, y)
            # Handle potential overflow or invalid math results
            if math.isinf(predicted_result) or math.isnan(predicted_result):
                return float('inf') # Assign a very high error
            total_error += abs(predicted_result - expected_result)
        except (ValueError, OverflowError):
            return float('inf') # Penalize expressions that cause errors
            
    return total_error


def select_parents_roulette_wheel(population, fitnesses):
    """
    Selects a parent using fitness proportionate (roulette wheel) selection.
    Individuals with lower error (better fitness) have a higher chance of being selected.
    
    Args:
        population (list): The list of individuals.
        fitnesses (list): The corresponding fitness scores for the population.

    Returns:
        GPTree: The selected parent individual.
    """
    # Since lower fitness is better, we invert the scores.
    # A small epsilon is added to avoid division by zero if fitness is 0.
    inverted_fitnesses = [1.0 / (f + 1e-9) for f in fitnesses]
    total_inverted_fitness = sum(inverted_fitnesses)

    # If all fitnesses are infinite, pick randomly to avoid errors
    if total_inverted_fitness == 0:
        return random.choice(population).copy()
        
    # Pick a random point on the wheel
    pick = random.uniform(0, total_inverted_fitness)
    current = 0
    for i, individual in enumerate(population):
        current += inverted_fitnesses[i]
        if current > pick:
            return individual.copy()
    
    # Fallback in case of floating point inaccuracies
    return population[-1].copy()


def crossover(parent1, parent2):
    """
    Performs subtree crossover on two parents to create one offspring.
    A random subtree from parent2 replaces a random subtree in a copy of parent1.

    Args:
        parent1 (GPTree): The first parent.
        parent2 (GPTree): The second parent.

    Returns:
        GPTree: The new offspring.
    """
    offspring = parent1.copy()
    
    # Collect all nodes from both parents
    nodes1 = offspring.collect_nodes()
    nodes2 = parent2.collect_nodes()
    
    # Select random nodes to be the crossover points
    crossover_point1 = random.choice(nodes1)
    crossover_point2 = random.choice(nodes2).copy() # Copy the subtree
    
    # Swap the subtrees by replacing data and children pointers
    crossover_point1.data = crossover_point2.data
    crossover_point1.children = crossover_point2.children
    
    return offspring


def mutation(individual):
    """
    Performs subtree mutation on an individual.
    A random node is chosen, and its subtree is replaced with a new random one.
    
    Args:
        individual (GPTree): The individual to mutate.
    """
    nodes = individual.collect_nodes()
    mutation_point = random.choice(nodes)
    
    # Generate a new random subtree with a limited depth
    new_subtree = generate_individual(max_depth=3, method='grow')
    
    # Replace the old subtree with the new one
    mutation_point.data = new_subtree.data
    mutation_point.children = new_subtree.children


# -----------------------------------------------------------------------------
# 4. VISUALIZATION
# -----------------------------------------------------------------------------

def draw_tree(tree, filename="best_solution_tree"):
    """
    Visualizes an expression tree using graphviz and saves it to a file.
    
    Args:
        tree (GPTree): The tree to visualize.
        filename (str): The base name for the output file.
    """
    if not GRAPHVIZ_AVAILABLE:
        print("Skipping tree visualization.")
        return

    dot = Digraph(comment='Best Evolved Expression')
    
    # Inner function to recursively add nodes and edges to the graph
    def add_nodes_edges(t, dot, node_id=0):
        node_name = f"node{node_id}"
        dot.node(node_name, t.get_node_name())
        
        child_start_id = node_id + 1
        for child in t.children:
            child_node_name = f"node{child_start_id}"
            dot.edge(node_name, child_node_name)
            child_start_id = add_nodes_edges(child, dot, child_start_id)
            
        return child_start_id

    add_nodes_edges(tree, dot)
    dot.render(filename, format='png', cleanup=True)
    print(f"Best solution tree saved to '{filename}.png'")


# -----------------------------------------------------------------------------
# 5. MAIN EVOLUTIONARY LOOP
# -----------------------------------------------------------------------------

def evolve():
    """
    The main function that runs the Genetic Programming evolutionary loop.
    """
    print("--- Starting Genetic Programming for Symbolic Regression ---")
    
    # Step 1: Initialize Population
    population = init_population()
    
    best_overall_individual = None
    best_overall_fitness = float('inf')

    # Step 5: Start the Evolutionary Loop
    for gen in range(MAX_GENERATIONS):
        
        # Step 5.1: Evaluate Fitness
        fitnesses = [calculate_fitness(ind, DATASET) for ind in population]

        # Keep track of the best individual found so far
        min_fitness_in_gen = min(fitnesses)
        if min_fitness_in_gen < best_overall_fitness:
            best_overall_fitness = min_fitness_in_gen
            best_individual_index = fitnesses.index(best_overall_fitness)
            best_overall_individual = population[best_individual_index].copy()
            print(f"Generation {gen:03d}: New Best Fitness (Error) = {best_overall_fitness:.4f}")

        # Step 5.5: Check Termination Condition
        if best_overall_fitness < TERMINATION_FITNESS:
            print("\nTermination condition met: Found a sufficiently good solution.")
            break

        # Step 5.2-5.4: Create the Next Generation
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            # Select parents
            parent1 = select_parents_roulette_wheel(population, fitnesses)
            
            if random.random() < CROSSOVER_RATE:
                parent2 = select_parents_roulette_wheel(population, fitnesses)
                child = crossover(parent1, parent2)
            else:
                child = parent1 # Crossover didn't happen, use parent1 as basis
            
            if random.random() < MUTATION_RATE:
                mutation(child)

            new_population.append(child)
        
        population = new_population

    # Step 6: Final Report
    print("\n\n--- Evolution Finished ---")
    if best_overall_individual:
        print(f"Best Individual's Fitness (Final Error): {best_overall_fitness:.4f}")
        print(f"Evolved Expression (Prefix Notation): {best_overall_individual}")
        draw_tree(best_overall_individual)
        print("Succesfull\n")
    else:
        print("No valid solution was found.")


if __name__ == "__main__":
    evolve()