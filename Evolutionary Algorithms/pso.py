"""
* File:    pso.py
* Purpose: This module contains the declaration and implementation of the
           particle swarm optimization algorithm for knapsacks.

           Example Use:
              from pso.py import *
              from knapsack_generator.py import *

              # create a knapsack generator
              knapsack_gen = knapsack_generator(None, 100, 100)

              # create a knapsack problem from the generator
              knapsack_problem = knapsack_gen.next()
              
          To run canonical pso algorithm:
              knapsack_contents = pso(knapsack_problem)

          The search algorithm returns a list of contents represented by
          {0, 1} where a '1' indicates that the item is in the knapsack, and
          '0' otherwise.

"""

from copy import *
from random import *
from knapsack import *

# class representing PSO particle for a knapsack problem
class particle(object):
    def __init__(self, knapsack, index, swarm):
        self.knapsack = knapsack
        self.index = index
        self.pbest = knapsack
        self.swarm = swarm
        self.cognition = swarm.population_cognition
        self.social = swarm.population_social
        self.inertial = swarm.population_inertial
        self.velocity = 0
        self.max_velocity = len(self.knapsack.content_list)-1

    # f(n) that determines fitness of an individual
    # Maximizes the values of the knapsack contents by returning the value
    # of the items in the pack, unless the weight violates the knapsack
    # capacity contraint.
    def fitness(self, knap):
        capacity = 0
        value = 0
        for i in range(0, len(knap.content_list)):
            if knap.content_list[i] == 1:
                capacity += knap.weights[i]
                value += knap.values[i]

        if capacity > knap.capacity:
            return 0
        else:
            return value

    def update_velocity(self):

        #print 'update velocity knapsack['+str(self.index)+']'

        # Note:
        #       Using the fitness f(n) that returns a knapsack value returns
        #       high numbers, i.e. > 150, and more fit the knapsack soln is,
        #       the higher it is.  But we need this to be proportionate to the
        #       length of the content_list.

        global_distance = (self.fitness(self.swarm.global_best.knapsack) - 
                           self.fitness(self.knapsack))
        personal_distance = (self.fitness(self.pbest) - 
                             self.fitness(self.knapsack))

        # create a global ratio of fitness
        if self.fitness(self.knapsack) != 0 and \
        (self.fitness(self.swarm.global_best.knapsack))!= 0:
            global_ratio = 1 - (float(self.fitness(self.knapsack)) / 
                                (self.fitness(self.swarm.global_best.knapsack)))
        else:
            global_ratio = 1
        
        # create a personal ratio
        if  self.fitness(self.knapsack) != 0 and \
        (self.fitness(self.knapsack)) != 0:
            personal_ratio = 1 - (float(self.fitness(self.knapsack)) / 
                                  (self.fitness(self.pbest)))
        else:
            personal_ratio = 1

        # clamp the ratios.  Ratio will be > 1 when best is worse than current
        if global_ratio > 1:
            global_ratio = 0.1
        if personal_ratio > 1:
            personal_ratio = 0.1

        # create a relationship between the distance and the knapsack by
        # taking a percentage of the global best's knapsack
        # This creates a velocity to be used as a splicing point that is a 
        # percentage of the
        # global best
        self.velocity = self.max_velocity * global_ratio

        # adjust the velocity by the intertial constant
        self.velocity = self.inertial * self.velocity 

        # calculate the cognition
        cognition_distance = self.velocity * personal_ratio
        cognition_calc = abs((self.cognition * random()) * (cognition_distance))

        # adjust for sociality
        social_distance = self.velocity
        social_calc =  abs((self.social * random()) * (social_distance))

        self.velocity += (cognition_calc + social_calc)

        # make certain we get an integer to use as an index
        self.velocity = int(round(self.velocity))
        
        # clamp it to within the content_list
        if self.velocity > self.max_velocity:
            self.velocity = self.max_velocity
        elif self.velocity < 0:
            self.velocity = 0

    # particle method that updates the particle's position within the swarm
    def update_position(self):

        # pbest is pointing to knapsack, so before we update knapsack, 
        # make certain pbest retaints its
        # values from knapsack
        new_pbest = knapsack(self.knapsack.capacity, \
                             self.knapsack.weights, \
                             self.knapsack.values)
        new_pbest.content_list = self.knapsack.content_list[:]
        self.pbest = new_pbest

        # take the velocity and use it to splice the content of the swarm's 
        # best with our contents
        new_knapsack = knapsack(self.knapsack.capacity, \
                                self.knapsack.weights, \
                                self.knapsack.values)
        new_knapsack.content_list = \
        self.swarm.global_best.knapsack.content_list[:self.velocity] + \
        self.knapsack.content_list[self.velocity:]

        # now set the knapsack to the newly created knapsack
        self.knapsack = new_knapsack

        # update the local best if better
        if self.fitness(self.knapsack) > self.fitness(self.pbest):
            self.pbest = self.knapsack
        # update the global best if its better
        if self.fitness(self.knapsack) > \
        self.fitness(self.swarm.global_best.knapsack):
            self.swarm.global_best = self

# class representing PSO
class particle_swarm_optimization(object):

    def __init__(self, knapsack_problem, population_size, population_cognition,\
                 population_social, population_inertial, max_iterations):
        self.knapsack_problem = knapsack_problem
        self.population_size = population_size
        # cognition affects a particle's pbest influence on its movement
        self.population_cognition = population_cognition
        # social affects the swarms gbest influence on its movement
        self.population_social = population_social
        # intertial affects how much of the current partical velocity to account
        self.population_inertial = population_inertial
        self.population = []
        self.global_best = None
        self.max_iterations = max_iterations

        # create the initial swarm
        for i in range(0, population_size):
            # create a knapsack instance
            new = knapsack(self.knapsack_problem.capacity, \
                           self.knapsack_problem.weights, \
                           self.knapsack_problem.values)
            new.addrandomcontents()

            # create the swarm of particles
            new_particle = particle(new, i, self)
            self.population.append(new_particle)

            # randomize the order of population
            shuffle(self.population)

        # create the initial global best
        self.global_best = self.population[0]

    # put the swarm in motion within the solution space
    def fly(self):
        for i in range(self.max_iterations):
            for particle in self.population:
                # update the particle's velocity
                particle.update_velocity()
                # update the particle's position
                particle.update_position()


# Module Method that implements particle swarm optimization for knapsacks
# @param1: knapsack_problem generated from knapsack_generator
# @returns: returns list of values for x(i), where x(i) is an elem of {0, 1}
# Note:
def pso(knapsack_problem):

    # population size
    population_size = 100

    population_cognition = 2
    population_social = 2
    population_inertial = 0.95
    max_iterations = 100

    swarm = particle_swarm_optimization(knapsack_problem, population_size, \
                                        population_cognition,\
                                        population_social, population_inertial,\
                                        max_iterations)

    swarm.fly()

    return swarm.global_best.knapsack.content_list
