# problem.py
"""
This module defines the core components of the symbolic regression problem.
It includes the function set, terminal set, and the dataset loader.
This separation allows the problem to be easily modified without changing
the GP algorithm's core logic.
"""

import csv
import operator

# --- Helper Functions for the Problem ---

# The safe_div function is a crucial part of the 'Function Set'.
# It ensures 'evaluation safety' by preventing division-by-zero errors,
# which would otherwise crash the program during fitness evaluation.
def safe_div(x, y):
    """Protected division to avoid division-by-zero errors."""
    return x / y if abs(y) > 1e-6 else 1.0

# --- Problem Definition ---

# The Function Set defines the internal nodes available for building the expression trees.
# These are the basic operations the GP algorithm can use to form solutions.
FUNCTIONS = {
    '+': (operator.add, 2),
    '-': (operator.sub, 2),
    '*': (operator.mul, 2),
    '/': (safe_div, 2)
}

# The Terminal Set defines the leaf nodes of the expression trees.
# It includes the problem's independent variables ('x', 'y') and provides a
# source for ephemeral random constants.
TERMINALS = ['x', 'y']
CONSTANT_RANGE = (-5.0, 5.0)

# --- Dataset Loading ---
def load_dataset_from_csv(filename):
    """
    Loads the dataset from a specified CSV file.
    These data points are the 'fitness cases' against which each
    individual program is evaluated.
    """
    dataset = []
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            for line_num, row in enumerate(reader, 2):
                try:
                    if len(row) != 3:
                        print(f"Warning: Skipping malformed row {line_num}.")
                        continue
                    dataset.append(tuple(map(float, row)))
                except ValueError:
                    print(f"Warning: Skipping row {line_num} due to data conversion error.")
                    continue
        print(f"Successfully loaded {len(dataset)} data points from '{filename}'.")
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    return dataset

# The DATASET variable holds all fitness cases for the entire run.
DATASET = load_dataset_from_csv('symbolic_regression_data.csv')