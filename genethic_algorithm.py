import random
from math import exp
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import os
import imageio

def fitness_function_graphic():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x = np.arange(-2, 2, 0.01)
    y = np.arange(-2, 2, 0.01)
    x, y = np.meshgrid(x, y)
    z = np.exp(-1 * x**2 -1 * y**2 )# z = np.cos(x)*np.cos(y)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


class GenethicAlgorithm:

    def __init__(self, population_size, n_vector, limit_value_bottom, limit_value_top, p_mutation, bounds):

        self.POPULATION_SIZE        = population_size
        self.N_VECTOR               = n_vector
        self.LIMIT_VALUE_BOTTOM     = limit_value_bottom
        self.LIMIT_VALUE_TOP        = limit_value_top
        self.P_MUTATION             = p_mutation

        self.bounds_ = bounds
        
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
                    \nПопуляция: {[[round(gen, 5) for gen in individual] for individual in self.population]}\
                    \nМаксимальное значение: {round(max_fitness, 5)}\
                    \nМинимальное значение: {round(min_fitness, 5)}\
                    \nСреднее значение: {round(average_fitness, 5)}\n\n")
            if generation_count <= 10 or generation_count % 10 == 0:
                self.visualize_population(generation_count)
            
    def visualize_population(self, generation_number):
        if not os.path.exists('results/'):
            os.makedirs('results/')

        x0, x1 = self.bounds_[:2]
        x = np.arange(x0, x1 + 0.1, 0.1)
        y0, y1 = self.bounds_[2:]
        y = np.arange(y0, y1 + 0.1, 0.1)
        x, y = np.meshgrid(x, y)
        z = np.exp(-1 * x**2 -1 * y**2)

        plt.title(f"Поколение {generation_number}.")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.pcolormesh(x, y, z, cmap=cm.plasma)
        for i in range(len(self.population)):
            x_p, y_p = self.population[i][0], self.population[i][1]
            plt.plot(x_p, y_p, marker='D', linestyle='', color='lime')
        plt.savefig('results/' + str(generation_number) + '.png')
        plt.clf()
            

    def create_gif_visualisation(self):
        results = []
        filenames = os.listdir("results/")
        for filename in sorted(filenames, key=lambda x: int(os.path.splitext(x)[0])):
            results.append(imageio.imread("results/" + filename))
        imageio.mimsave('final_result.gif', results, duration=0.2)