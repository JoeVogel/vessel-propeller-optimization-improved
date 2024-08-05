import argparse
import numpy as np

from hgso import HGSO
from evolutionary_strategy import EvolutionaryStrategy, Solver

def run(instance, alpha, beta, epxilon, K, num_population, num_clusters):

    lower_bounds    = [0.5, 0.3, 0.5]
    upper_bounds    = [0.8, 1.05, 1.4]
    
    es = EvolutionaryStrategy(Solver.HGSO, service_speed=7, number_of_blades=5)
    obj_func = es.fitness_function

    hgso = HGSO(obj_func=obj_func, lb=lower_bounds, ub=upper_bounds, alpha=alpha, beta=beta, epxilon=epxilon, K=K, verbose=False, pop_size=num_population, n_clusters=num_clusters, random_seed=10)

    max_iter = 31
    for i in range(1, max_iter):
        hgso.solve(i)

    fitness = hgso.g_best[1]
    return fitness