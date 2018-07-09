"""
* File:    utils.py
* Purpose: This module contains the declaration and implementation of the
           knapsack local search algorithms.

           Example Use:
              from knapsack.py import *
              from knapsack_generator.py import *

              # create a knapsack generator
              knapsack_gen = knapsack_generator(None, 100, 100)

              # create a knapsack problem from the generator
              knapsack_problem = knapsack_gen.next()
              
          To run steepest_ascent hill climbing search algorithm:
              knapsack_contents = steepest_ascent(knapsack_problem)

          To run steepest_ascent with random restart hill climbing search
          algorithm:
              knapsack_contents = steepest_ascent_restart(knapsack_problem)

          The search algorithms return a list of contents represented by
          {0, 1} where a '1' indicates that the item is in the knapsack, and
          '0' otherwise.

"""


# Module Method that implements steepest_ascent local search
# @param1: knapsack_problem generated from knapsack_generator
# @returns: returns list of values for x(i), where x(i) is an elem of {0, 1}
def steepest_ascent(knapsack_problem):

    # create an empty knapsack initially as our best result
    best = knapsack(knapsack_problem.capacity, knapsack_problem.weights, \
                    knapsack_problem.values)

    while True:
        # create a new knapsack & add a random assortment of items to it
        new = knapsack(knapsack_problem.capacity, knapsack_problem.weights, \
                       knapsack_problem.values)
        new.addrandomcontents()

        # if the value of the contents of the newly created knapsack is not
        # as good as the current best, then return the best already found
        if new.content_value <= best.content_value:
            return best.content_list
        else:
            best = new


# Module Method that implements steepest_ascent_restart local search
# @param1: knapsack_problem generated from knapsack_generator
# @returns: returns list of values for x(i), where x(i) is an elem of {0, 1}
def steepest_ascent_restart(knapsack_problem):

    # create an empty knapsack initially as our best result
    best = knapsack(knapsack_problem.capacity, knapsack_problem.weights, \
                    knapsack_problem.values)

    for restart_iteration in range(0, 100):

        # run steepest_ascent local search
        # this will produce a random start state for this iteration
        resultlist = steepest_ascent(knapsack_problem)
        
        # create a new knapsack from the steepest_ascent results
        new = knapsack(knapsack_problem.capacity, knapsack_problem.weights, \
                       knapsack_problem.values)
        new.content_list = resultlist
        
        # update the value and the weight of the new knapsack
        for i in range(0, len(new.content_list)):
            if new.content_list[i] == 1:
                new.content_value += new.content_list[i]
                new.load += new.content_list[i]

        # if the value of the contents of the newly created knapsack is 
        # better than the current best, then cache it away
        if new.content_value > best.content_value:
            best = new

    return best.content_list 
