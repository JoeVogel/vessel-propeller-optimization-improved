import argparse
import numpy as np

from openai import OpenAIES
from evolutionary_strategy import EvolutionaryStrategy, Solver

def run(instance, sigma_init, sigma_decay, learning_rate, learning_rate_decay, weight_decay, num_population):

    lower_bounds    = [0.5, 0.3, 0.5]
    upper_bounds    = [0.8, 1.05, 1.4]
    
    es = EvolutionaryStrategy(Solver.OPENAI_ES, service_speed=7, number_of_blades=5)
    obj_func = es.fitness_function

    openai = OpenAIES(num_params=3,
                      sigma_init=sigma_init, 
                      sigma_decay=sigma_decay, 
                      learning_rate=learning_rate, 
                      learning_rate_decay=learning_rate_decay, 
                      weight_decay=weight_decay, 
                      popsize=num_population, 
                      lower_bounds=lower_bounds, 
                      upper_bounds=upper_bounds)
    
    min_fitness = 1000.0

    max_iter = 30
    for _ in range(max_iter):
        population = openai.ask()
            
        fitness_list = np.zeros(num_population)
        
        for j in range(len(population)):
            fitness_list[j] = es.fitness_function(population[j])
            
        openai.tell(fitness_list)

        if min(fitness_list) < min_fitness:
            min_fitness = min(fitness_list)
                   
    # print(min_fitness)
    return min_fitness
