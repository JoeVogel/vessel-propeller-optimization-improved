import argparse
import numpy as np
import sys
import os
import json

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from optimization.hgso import HGSO
from data_loader import load_hyperparameters
from optimization.evolutionary_strategy import EvolutionaryStrategy, Solver
from cost_measurement.utils import measure_resources

def run_iteration(iteration, hgso, obj_func):
    hgso.solve(iteration)

def run(alpha, beta, epxilon, K, num_population, num_clusters):

    lower_bounds    = [0.5, 0.3, 0.5]
    upper_bounds    = [0.8, 1.05, 1.4]
    
    es = EvolutionaryStrategy(Solver.HGSO, service_speed=7, number_of_blades=5)
    obj_func = es.fitness_function

    hgso = HGSO(obj_func=obj_func, lb=lower_bounds, ub=upper_bounds, alpha=alpha, beta=beta, epxilon=epxilon, K=K, verbose=False, pop_size=num_population, n_clusters=num_clusters, random_seed=10)

    max_iter = 31
    measure_iteration, resources = measure_resources(run_iteration, hgso, obj_func)
    
    for i in range(1, max_iter):
        measure_iteration()
        
    return hgso.g_best[1], resources

if __name__ == '__main__':
    
    solver  = Solver.HGSO
    
    configs = load_hyperparameters(solver)

    alpha           = configs['alpha']
    beta            = configs['beta']
    epsilon         = configs['epsilon']
    K               = configs['K']
    num_clusters    = configs['num_clusters']
    num_pop         = configs['num_population']

    fitness, resources = run(alpha, beta, epsilon, K, num_pop, num_clusters)

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

    with open('src/cost_measurement/results/hgso_stats.json', 'w') as json_file:
        json.dump(stats, json_file, indent=4)