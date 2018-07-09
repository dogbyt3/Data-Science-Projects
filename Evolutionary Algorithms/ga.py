"""
* File:    ga.py
* Purpose: This module contains the declaration and implementation of the
           particle swarm optimization algorithm for knapsacks.

           Example Use:
              from ga.py import *
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

from knapsack import *

# class representing an abstract genetic_algorithm
# This class is inherited by canonical_ga, and by constraint_ga
# the fitness f(n) must be overridden by the concrete impls
class genetic_algorithm(object):

    def __init__(self, knapsack_problem, generations, initial_population):
        self.knapsack_problem = knapsack_problem
        self.generations = generations
        self. initial_population = initial_population

        # create the initial population
        self.population = []
        for i in range(0, initial_population):
            new = knapsack(self.knapsack_problem.capacity, \
                           self.knapsack_problem.weights, \
                           self.knapsack_problem.values)
            new.addrandomcontents()
            self.population.append(new)

        # randomize the order of population
        shuffle(self.population)


    # f(n) to create a child from two parents by selecting
    # portions of each randomly.  This creates a random crossover
    # Note: Adapted from AIMA search.py code
    def reproduce(self, parent1, parent2):

        # select a random point within the content lists
        crossover_point = randrange(len(parent1.content_list))

        # create the initial child
        new = knapsack(self.knapsack_problem.capacity, \
                       self.knapsack_problem.weights, \
                       self.knapsack_problem.values)

        # using the random crossover point, concatenate the content lists of 
        # the parents into the child's
        new.content_list = parent1.content_list[:crossover_point] + \
        parent2.content_list[crossover_point:]
        return new

    # f(n) to mutate a knapsack's contents randomly
    def mutate(self, knapsack):

        # select the number of bits to modify
        r = randint(1, len(knapsack.content_list))
        mod = len(knapsack.content_list)%r

        # randomly change bits within the content_list
        for i in range(0, len(knapsack.content_list)):
            item = i + mod
            if item <= (len(knapsack.content_list)-1):
                if knapsack.content_list[item] == 1:
                    knapsack.content_list[item] == 0
                else:
                    knapsack.content_list[item] == 1


    # f(n) that determines fitness of an individual
    # note: method is abstact and meant to be overridden by concrete impls
    def fitness(self, knapsack):
        abstract

    # f(n) that returns a weighted selection of an individual within a 
    # population
    # Apply the fitness function to each individual within the population,
    # add up the total fitness, calculate the average fitness of the population
    # then choose two individuals from the population whose fitness is above the
    # average of the population.
    def random_selection(self, population):
        average_fitness = 0
        selections = []
        tmp_population = population[:]

        # ensure that we return a set of parents from the population
        while len(selections) < 2:
            # for each parent randomly select from the fittest in the population
            for individual in tmp_population:
                # calculate the average fitness of the selection population
                average_fitness = self.avg_population_fitness(tmp_population)
                # select a random index for an individual within the population
                r = randint(0, len(tmp_population)-1)
                # if this individual is fitter than average, add it to the 
                # parent list
                if self.fitness(tmp_population[r]) >= average_fitness:
                    # append this individual to the selection list and remove it
                    # from the temp population preventing selecting same 
                    # individual twice
                    selections.append(tmp_population[r])
                    tmp_population.pop(r)
                    break

        return selections

    # f(n) that returns the average fitness value for a population
    def avg_population_fitness(self, population):
        fitness_total = 0
        # calculate the average fitness of the population
        for individual in population:
            fitness_total += self.fitness(individual)
        return fitness_total/len(population)
          
    # search method that finds the fittest knapsack solution and returns it.
    def search(self):
        for i in range(self.generations):
            new_population = []

            for j in range(0, len(self.population)):
                # randomly select 2 parents from the population
                parent1, parent2 = self.random_selection(self.population)

                # create a new child by reproduction
                child = self.reproduce(parent1, parent2)

                # mutate the child, with some small random probability
                # this probability was arbitrarily chosen to be small.
                # can be tweaked to increase or decrease mutation probability
                if uniform(0, 1) > 0.8:
                    self.mutate(child)
                new_population.append(child)

            # create the new population G'
            self.population = new_population

        # of all the generations, find the fittest individual & return it
        best = self.population[0]
        for i in range(0, len(self.population)):
            if self.fitness(self.population[i]) > best:
                best = self.population[i]

        return best.content_list


# This class represents the classic Genetic Algorithm implementation
# canonical_ga inherits from the abstract base class genetic_algorithm
class canonical_ga(genetic_algorithm):
    
    def __init__(self, knapsack_problem, generations, initial_population):
        genetic_algorithm.__init__(self, knapsack_problem, generations, \
                                   initial_population)

    # f(n) that determines fitness of an individual
    # Maximizes the values of the knapsack contents by returning the value
    # of the items in the pack, unless the weight violates the knapsack
    # capacity contraint.
    def fitness(self, knapsack):
        capacity = 0
        value = 0
        for i in range(0, len(knapsack.content_list)):
            if knapsack.content_list[i] == 1:
                capacity += knapsack.weights[i]
                value += knapsack.values[i]

        if capacity > knapsack.capacity:
            return 0
        else:
            return value


# This class represents the Constraint-Violation Genetic Algorithm 
# implementation. 
# constraint_ga inherits from the abstract base class genetic_algorithm
class constraint_ga(genetic_algorithm):
    
    def __init__(self, knapsack_problem, generations, initial_population):
        genetic_algorithm.__init__(self, knapsack_problem, generations, \
                                   initial_population)

    # f(n) that determines fitness of an individual
    def fitness(self, knapsack):
        capacity = 0
        value = 0
        for i in range(0, len(knapsack.content_list)):
            if knapsack.content_list[i] == 1:
                capacity += knapsack.weights[i]
                value += knapsack.values[i]

        if capacity > knapsack.capacity:
            # calculate the penalty such that the further away from optimum
            # the larger the penalty applied to the fitness score
            penalty = value - (value/(capacity - knapsack.capacity))
            return value - penalty
        else:
            return value
            
# Module Method that implements standard genetic algorithm
# @param1: knapsack_problem generated from knapsack_generator
# @returns: returns list of values for x(i), where x(i) is an elem of {0, 1}

def ga_canonical(knapsack_problem):

    # number of generations
    generations = 100

    # number of individuals in initial population
    # create an initial population of 100 knapsack solutions
    population_count = 100

    ga = canonical_ga(knapsack_problem, generations, population_count)
    result = ga.search()

    return result



# Module Method that implements constraints violation genetic algorithm
# @param1: knapsack_problem generated from knapsack_generator
# @returns: returns list of values for x(i), where x(i) is an elem of {0, 1}
def ga_constraint(knapsack_problem):

    # number of generations
    generations = 100

    # number of individuals in initial population
    # create an initial population of 100 knapsack solutions
    population_count = 100

    cga = constraint_ga(knapsack_problem, generations, population_count)
    result = cga.search()

    return result

