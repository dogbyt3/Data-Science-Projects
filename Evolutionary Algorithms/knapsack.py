"""
* File:    knapsack.py
* Purpose: This module contains the declaration and implementation of the
           knapsack object
"""

# import knapsack_generator & knapsack_problem objects
from knapsack_generator import *

#import random to create random knapsack contents 
from random import *

# class representing a knapsack instance
class knapsack(object):

    def __init__(self, capacity, weights, values):
        self.capacity = capacity
        self.weights = weights
        self.values = values
        self.load = 0
        self.content_value = 0
        self.content_list = []

        # set initial content list to all 0's
        for i in range(len(self.weights)):
            self.content_list.append(0)

    # method that adds random contents to the knapsack
    # ensures a valid knapsack given constraints of weight and 0/1
    # for items.  Items are unique and either in the knapsack, or not.
    def addrandomcontents(self):

         # randomly select an item and place it in the bag if there is room
         random_index = randint(0, len(self.content_list)-1)

         contents_exhausted = False
         # add random contents until no more can be added to the knapsack
         while not contents_exhausted:

             # only add this item if it has not already been added
             if self.content_list[random_index] != 1:
                 self.additem(random_index)

             # make sure there is capacity for the remaining items
             for i in range(0, len(self.content_list)):
                 # only those items that have not been already added
                 if self.content_list[i] == 0:
                     if self.load + self.weights[i] <= self.capacity:
                         # found at least one item left that will fit
                         # break out of this loop
                         # leaving i < len(content_list)-1
                         contents_exhausted = False
                         break
    
             # if we have iterated over the remaining items and did not
             # find an item to place in the knapsack, then we are done
             if i == (len(self.content_list)-1):
                 contents_exhausted = True
             else:
                 random_index = randint(0, len(self.content_list)-1)


    # add an item to the knapsack checking to see if it will fit first
    # update the weight, and values of the knapsack instance
    def additem(self, index):
        if (self.load + self.weights[index]) <= self.capacity:
            self.load += self.weights[index]
            self.content_value += self.values[index]
            self.content_list[index] = 1
            