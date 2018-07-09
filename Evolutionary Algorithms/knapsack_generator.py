"""
* File:    knapsack_generator.py
* Purpose: This module contains the declaration and implementation of the
           knapsack generator object
"""

import random

class knapsack_generator (object) :

    def __init__(self, seed = None, n = 100, capacity = 100) :

        """
        parameters:
        seed - an integer value if you want the sequence of problems generated
        to be reproducible
        n - number of variables
        capacity - the capacity of the knapsack
        """

        self.n = n
        self.capacity = capacity
        self.maxWeight = int(capacity / 5)
        self.maxValue = int(capacity / 5)
        self.random = random.Random(seed)

    def next(self) :

        weights = [self.random.randint(1, self.maxWeight)
                   for i in range(self.n)]
        values = [self.random.randint(1, self.maxValue)
                  for i in range(self.n)]

        return knapsack_problem(self.capacity, weights, values)

class knapsack_problem (object) :

    def __init__(self, capacity, weights, values) :

        self.capacity = capacity
        self.weights = weights
        self.values = values

    def __len__(self) :

        return len(self.weights)
        
