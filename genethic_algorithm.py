import random
from math import exp
from copy import copy

class GenethicAlgorithm:

    def __init__(self, population_size, n_vector, limit_value_bottom, limit_value_top, p_mutation):

        self.POPULATION_SIZE        = population_size
        self.N_VECTOR               = n_vector
        self.LIMIT_VALUE_BOTTOM     = limit_value_bottom
        self.LIMIT_VALUE_TOP        = limit_value_top
        self.P_MUTATION             = p_mutation
        
        self.minimum_fitness_values = []
        self.maximum_fitness_values = []
        self.average_fitness_values = []

        self.population = self.create_population(self.POPULATION_SIZE)

        fitness_values = list(map(self.fitness_function, self.population))

        for individual, fitnessValue in zip(self.population, fitness_values):
            individual.value = fitnessValue

        self.population.sort(key=lambda ind: ind.value)

        print([str(ind) + ", " + str(ind.value) for ind in self.population])
        print('\n\n')


    class Individual(list):
        def __init__(self, *args):
            super().__init__(*args)
            self.value = 0

    def fitness_function(self, f):
        return exp(-1 * f[0]**2 -1 * f[1]**2 ) #exp(- x^2 - y^2)

    def create_individual(self):
        ind = self.Individual([random.uniform(self.LIMIT_VALUE_BOTTOM, self.LIMIT_VALUE_TOP) for _ in range(self.N_VECTOR)])
        ind.value = self.fitness_function(ind)
        return ind

    def create_population(self, n):
        return list([self.create_individual() for _ in range (n)])

    def selection(self, population):
        offspring = []

        for _ in range(self.POPULATION_SIZE):
            i1 = i2 = 0
            while i1 == i2:
                i1, i2 = random.randint(0, self.POPULATION_SIZE - 1), random.randint(0, self.POPULATION_SIZE - 1)
            
            offspring.append(copy(max(population[i1], population[i2], key=lambda ind: ind.value)))

        return offspring
    
    def crossbreeding(self, object1, object2):
        s = random.randint(1, len(object1) - 1)
        object1[:s], object2[:s] = object2[:s], object1[:s]

    def mutation(self, mutant):
        mutation_percent = 0.05
        for index in range(len(mutant)):

            if random.random() > self.P_MUTATION:
                continue

            s = 0
            while (s == 0):
                s = random.randint(-1, 1)
            mutant[index] += s * mutation_percent * mutant[index]

    def generation_algorithm(self, max_generation_count):

        generation_count = 0

        while generation_count < max_generation_count:
            generation_count += 1
            offspring = self.selection(self.population)

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                self.crossbreeding(child1, child2)
            
            for mutant in offspring:    
                self.mutation(mutant)
            
            fitness_values = list(map(self.fitness_function, offspring))
            for ind, fitness_value in zip(offspring, fitness_values):
                ind.value = fitness_value

            self.population = offspring

            min_fitness = min(fitness_values)
            max_fitness = max(fitness_values)
            average_fitness = sum(fitness_values) / len(fitness_values)

            self.minimum_fitness_values.append(min_fitness)
            self.maximum_fitness_values.append(max_fitness)
            self.average_fitness_values.append(average_fitness)

            print(  f"Поколение: {generation_count}\
                    \nПопуляция: {self.population}\
                    \nМаксимальное значение: {max_fitness}\
                    \nМинимальное значение: {min_fitness}\
                    \nСреднее значение: {average_fitness}\n\n")
