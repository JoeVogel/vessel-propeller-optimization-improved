import argparse
import numpy as np
import sys
import os
import json

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from optimization.openai import OpenAIES
from data_loader import load_hyperparameters
from optimization.evolutionary_strategy import EvolutionaryStrategy, Solver
from cost_measurement.utils import measure_resources

def run_iteration(iteration, openai, obj_func, num_population, min_fitness):
    population = openai.ask()
            
    fitness_list = np.zeros(num_population)
    
    for j in range(len(population)):
        fitness_list[j] = obj_func(population[j])
        
    openai.tell(fitness_list)

    if min(fitness_list) < min_fitness:
        min_fitness = min(fitness_list)
        
    return min_fitness

def run(sigma_init, sigma_decay, learning_rate, learning_rate_decay, weight_decay, num_population):

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
    measure_iteration, resources = measure_resources(run_iteration, openai, obj_func, num_population, min_fitness)
    
    for _ in range(max_iter):
        min_fitness = measure_iteration()
        
    
    return min_fitness, resources

if __name__ == '__main__':
   
    solver      = Solver.OPENAI_ES
    
    configs = load_hyperparameters(solver)

    sigma_init          = configs['sigma_init']
    sigma_decay         = configs['sigma_decay']
    learning_rate       = configs['learning_rate']
    learning_rate_decay = configs['learning_rate_decay']
    weight_decay        = configs['weight_decay']
    num_pop             = configs['num_population']

    fitness, resources = run(sigma_init, sigma_decay, learning_rate, learning_rate_decay, weight_decay, num_pop)

    df = pd.DataFrame(resources, columns=['Elapsed Time (s)', 'CPu Usage (%)', 'Memory Usage (bytes)'])

    df['Elapsed Time (s)']      = df['Elapsed Time (s)'].astype(float)
    df['CPu Usage (%)']         = df['CPu Usage (%)'].astype(float)
    df['Memory Usage (bytes)']  = df['Memory Usage (bytes)'].astype(int)

    mean_elapsed_time   = df['Elapsed Time (s)'].mean().round(3)
    mean_cpu            = df['CPu Usage (%)'].mean().round(2)
    mean_memory         = int(df['Memory Usage (bytes)'].mean())

    print("\n")
    print(f"Fitness:\t\t\t\t{fitness}")
    print(f"Iterations:\t\t\t\t{len(df)}")
    print(f"Mean EXECUTION TIME (s) by iteration:\t{mean_elapsed_time}")
    print(f"Mean CPU usage (%) by iteration:\t{mean_cpu}")
    print(f"Mean MEMORY usage (bytes) by iteration:\t{mean_memory}")
    print("\n")

    stats = {
        'Population size': num_pop,
        'Iterations': len(df),
        'Evaluated candidates': num_pop*len(df),
        'Mean iteration time (s)': mean_elapsed_time,
        'Mean time (s) by candidate (mean iteration time / population size)': mean_elapsed_time/num_pop,
        'Mean iteration cpu usage (%)': mean_cpu,
        'Mean iteration memory usage (bytes)': mean_memory
    }

    with open('src/cost_measurement/results/openai_stats.json', 'w') as json_file:
        json.dump(stats, json_file, indent=4)