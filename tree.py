# tree.py
"""
This module defines the core data structure for Genetic Programming: the
expression tree (GPTree). It also includes functions for creating and
initializing these trees, such as the ramped half-and-half method.
"""

import random
import copy
from problem import FUNCTIONS, TERMINALS, CONSTANT_RANGE

class GPTree:
    """
    Represents an individual program, which is the core 'representation' in GP.
    The tree structure naturally handles the hierarchy of mathematical expressions.
    """
    def __init__(self, data=None, children=None):
        self.data = data
        self.children = children if children else []

    def __str__(self):
        """Returns the expression in prefix notation for easy reading."""
        if not self.children:
            return str(self.data)
        return f"({self.get_node_name()} {' '.join(str(c) for c in self.children)})"

    def get_node_name(self):
        """Helper to get a clean string name for a function or terminal."""
        for name, (func, arity) in FUNCTIONS.items():
            if func == self.data:
                return name
        return str(self.data)

    def size(self):
        """Recursively counts nodes. Can be used to control for bloat."""
        return 1 + sum(child.size() for child in self.children)

    def collect_nodes(self):
        """Returns a flat list of all nodes for easy access by crossover/mutation."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.collect_nodes())
        return nodes

    def copy(self):
        """Creates a deep copy, essential for creating offspring without modifying parents."""
        return copy.deepcopy(self)


def generate_individual(max_depth, method='grow', current_depth=0):
    """
    Generates a single random tree, the basis of population initialization.
    - 'full' method: Creates bushy trees by picking only functions until max depth.
    - 'grow' method: Creates asymmetric trees by picking from both functions and terminals.
    """
    if current_depth >= max_depth or (method == 'grow' and random.random() < 0.5):
        # Base case: create a terminal (leaf) node.
        if random.random() < 0.5:
            terminal = random.choice(TERMINALS)
        else:
            terminal = round(random.uniform(*CONSTANT_RANGE), 2)
        return GPTree(data=terminal)

    # Recursive step: create a function (internal) node.
    func_name = random.choice(list(FUNCTIONS.keys()))
    func, arity = FUNCTIONS[func_name]
    
    children = [generate_individual(max_depth, method, current_depth + 1) for _ in range(arity)]
    return GPTree(data=func, children=children)


def init_population(pop_size, min_depth, max_depth):
    """
    Creates the initial population using the 'Ramped Half-and-Half' method.
    This is a standard GP technique to ensure the initial population has a
    high degree of structural diversity.
    """
    population = []
    num_per_depth = pop_size // (2 * (max_depth - min_depth + 1))
    
    for depth in range(min_depth, max_depth + 1):
        for _ in range(num_per_depth):
            population.append(generate_individual(depth, method='full'))
            population.append(generate_individual(depth, method='grow'))

    while len(population) < pop_size:
        depth = random.randint(min_depth, max_depth)
        population.append(generate_individual(depth, method='grow'))
        
    return population