import argparse
import numpy as np
import sys
import os
import json

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from optimization.pso import PSO
from data_loader import load_hyperparameters
from optimization.evolutionary_strategy import EvolutionaryStrategy, Solver
from cost_measurement.utils import measure_resources


def run_iteration(iteration, pso, obj_func):
    pso.solve()

def run(w, c1, c2, num_particles):

    lower_bounds    = [0.5, 0.3, 0.5]
    upper_bounds    = [0.8, 1.05, 1.4]
    
    es = EvolutionaryStrategy(Solver.PSO, service_speed=7, number_of_blades=5)
    obj_func = es.fitness_function

    pso = PSO(obj_func=obj_func, dimension=3, lower_bounds=lower_bounds, upper_bounds=upper_bounds, num_particles=num_particles, w=w, c1=c1, c2=c2, seed=0)
    
    max_iter = 30
    measure_iteration, resources = measure_resources(run_iteration, pso, obj_func)
    
    for _ in range(max_iter):
        measure_iteration()
        
    solution, fitness = pso.get_result()
    
    return fitness, resources


if __name__ == '__main__':
    solver  = Solver.PSO
    
    configs = load_hyperparameters(solver)
 
    c1      = configs['c1']
    c2      = configs['c2']
    weight  = configs['weight'] 
    num_pop = configs['population']

    fitness, resources = run(weight, c1, c2, num_pop)

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

    with open('src/cost_measurement/results/pso_stats.json', 'w') as json_file:
        json.dump(stats, json_file, indent=4)