"""
* File:    GA_vs_PSO_experiment.py
* Purpose: This module contains experiments comparing two evolutionary algorithms

           Example Use:
              from knapsack.py import *
              from knapsack_generator.py import *
              from ga import *
              from pso import *

              # create a knapsack generator
              knapsack_gen = knapsack_generator(None, 100, 100)

              # create a knapsack problem from the generator
              knapsack_problem = knapsack_gen.next()
              
          To run canonical genetic algorithm:
              knapsack_contents = ga_canonical(knapsack_problem)

          To run constraint-violation genetic algorithm:
              knapsack_contents = ga_constraint(knapsack_problem)

          To run canonical pso algorithm:
              knapsack_contents = pso(knapsack_problem)
                            
          * Not for this experiment, but could compare these to stochastic local search
          methods as well via:
          
          from utils import *
          
          To run steepest_ascent hill climbing search algorithm:
              knapsack_contents = steepest_ascent(knapsack_problem)

          To run steepest_ascent with random restart hill climbing search
          algorithm:
              knapsack_contents = steepest_ascent_restart(knapsack_problem)


          The search algorithms return a list of contents represented by
          {0, 1} where a '1' indicates that the item is in the knapsack, and
          '0' otherwise.

"""
from knapsack import *
from knapsack_generator import *
from pso import *
from ga import *
import time

if (__name__ == '__main__'):
    
    knapsack_gen = knapsack_generator(None, 100, 100)

    pso_total_weights = 0
    pso_total_values = 0
    pso_total_time = 0
    ga_total_weights = 0
    ga_total_values = 0
    ga_total_time = 0

    for iter in range(0, 5):

        knapsack_prob = knapsack_gen.next()
        print 'Iteration: '+ str(iter)
        print 'Knapsack capacity: '+ str(knapsack_prob.capacity)
        capacity = 0
        value = 0
        for i in range(0, len(knapsack_prob.weights)):
            capacity += knapsack_prob.weights[i]
            value += knapsack_prob.values[i]
        print 'Knapsack Total Weights: ' + str(capacity)+' Total Values: '+ \
        str(value)

        # pso
        print '' 
        print '#############                PSO               #################'
        print '#'
        start_time = time.time()
        result = pso(knapsack_prob)
        elapsed_time = (time.time()-start_time)

        capacity = 0
        value = 0
        for i in range(0, len(knapsack_prob.weights)):
            if result[i] == 1:
                capacity += knapsack_prob.weights[i]
                value += knapsack_prob.values[i]
        print '# Result Total Weights: ' + str(capacity)+' Total Values: '+ \
        str(value) +' Exec Time: '+str(elapsed_time)
        print '#'
        print '################################################################'
        print ''
        pso_total_weights += capacity
        pso_total_values += value
        pso_total_time += elapsed_time


        # ga_feasible
        print ''
        print '#############        Std Genetic Algorithm      #################'
        print '#'
        start_time = time.time()
        result = ga_canonical(knapsack_prob)
        elapsed_time = (time.time()-start_time)

        capacity = 0
        value = 0
        for i in range(0, len(knapsack_prob.weights)):
            if result[i] == 1:
                capacity += knapsack_prob.weights[i]
                value += knapsack_prob.values[i]
        print '# Result Total Weights: ' + str(capacity)+' Total Values: '+ \
        str(value) +' Exec Time: '+str(elapsed_time)
        print '#'
        print '################################################################'
        print ''
        ga_total_weights += capacity
        ga_total_values += value
        ga_total_time += elapsed_time

    print 'PSO avg weight: '+ str(pso_total_weights /(iter +1))
    print 'PSO avg value: '+ str(pso_total_values /(iter +1))
    print 'PSO avg time: '+ str(pso_total_time /(iter +1))

    print 'GA avg weight: '+ str(ga_total_weights /(iter +1))
    print 'GA avg value: '+ str(ga_total_values /(iter +1))
    print 'GA avg time: '+ str(ga_total_time /(iter +1))
