import matplotlib.pyplot as plt
from genethic_algorithm import *

POPULATION_SIZE    =  4
MAX_GENERATION     =  100
P_MUTATION         =  0.5
N_VECTOR           =  2
LIMIT_VALUE_BOTTOM = -2.2
LIMIT_VALUE_TOP    =  2.2

if __name__ == "__main__":
    fitness_function_graphic()
    genethic_algorithm_instance = GenethicAlgorithm(
        population_size    = POPULATION_SIZE, 
        n_vector           = N_VECTOR, 
        limit_value_bottom = LIMIT_VALUE_BOTTOM, 
        limit_value_top    = LIMIT_VALUE_TOP,
        p_mutation         = P_MUTATION,
        bounds             = [-2.2, 2.2, -2.2, 2.2]
    )

    genethic_algorithm_instance.generation_algorithm(MAX_GENERATION)

    minimum_fitness_values = genethic_algorithm_instance.minimum_fitness_values
    maximum_fitness_values = genethic_algorithm_instance.maximum_fitness_values
    average_fitness_values = genethic_algorithm_instance.average_fitness_values

    plt.plot(minimum_fitness_values[int(MAX_GENERATION * 0.1):], color='red')
    plt.plot(average_fitness_values[int(MAX_GENERATION * 0.1):], color='green')
    plt.plot(maximum_fitness_values[int(MAX_GENERATION * 0.1):], color='blue')
    plt.xlabel('Поколение')
    plt.ylabel('Мин/средняя/max приспособленность')
    plt.title('Зависимость min, average, max приспособленности от поколения')
    plt.show()

    genethic_algorithm_instance.create_gif_visualisation()