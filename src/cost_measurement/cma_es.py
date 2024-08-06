import argparse
import numpy as np
import sys
import os
import json

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import cma
from data_loader import load_hyperparameters
from optimization.evolutionary_strategy import EvolutionaryStrategy, Solver
from cost_measurement.utils import measure_resources

def run_iteration(iteration, cma_es, obj_func):
    population = cma_es.ask()
    cma_es.tell(population, [obj_func(individual) for individual in population])
    fitness = cma_es.result.fbest
    return fitness

def run(sigma, num_population):

    lower_bounds    = [0.5, 0.3, 0.5]
    upper_bounds    = [0.8, 1.05, 1.4]

    np.random.seed(10)
            
    x0 =  [
            np.random.uniform(lower_bounds[0], upper_bounds[0]),
            np.random.uniform(lower_bounds[1], upper_bounds[1]),
            np.random.uniform(lower_bounds[2], upper_bounds[2])
        ]
    
    es = EvolutionaryStrategy(Solver.CMA_ES, service_speed=7, number_of_blades=5)
    obj_func = es.fitness_function

    cma_es = cma.CMAEvolutionStrategy(x0,
                                      sigma,
                                      {'verbose':-9,
                                        'popsize': num_population,
                                        'bounds': [lower_bounds, upper_bounds]})
    
    max_iter = 30
    measure_iteration, resources = measure_resources(run_iteration, cma_es, obj_func)
    
    for _ in range(max_iter):
            
        measure_iteration()
        
    fitness = cma_es.result.fbest
    
    return fitness, resources


if __name__ == '__main__':
    
    configs = load_hyperparameters(Solver.CMA_ES)
    
    sigma = configs['sigma']
    num_pop = configs['num_population']

    fitness, resources = run(sigma, num_pop)

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

    with open('src/cost_measurement/results/cma_stats.json', 'w') as json_file:
        json.dump(stats, json_file, indent=4)