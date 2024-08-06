import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import cma
from optimization.evolutionary_strategy import EvolutionaryStrategy, Solver

def run(instance, sigma, num_population):

    lower_bounds    = [0.5, 0.3, 0.5]
    upper_bounds    = [0.8, 1.05, 1.4]

    np.random.seed(10)
            
    x0 =  [
            np.random.uniform(lower_bounds[0], upper_bounds[0]),
            np.random.uniform(lower_bounds[1], upper_bounds[1]),
            np.random.uniform(lower_bounds[2], upper_bounds[2])
        ]
    
    es = EvolutionaryStrategy(Solver.CMA_ES, service_speed=8.5, number_of_blades=6)
    obj_func = es.fitness_function

    cma_es = cma.CMAEvolutionStrategy(x0,
                                      sigma,
                                      {'verbose':-9,
                                        'popsize': num_population,
                                        'bounds': [lower_bounds, upper_bounds]})
    
    max_iter = 30
    for _ in range(max_iter):
            
        population = cma_es.ask()
        cma_es.tell(population, [obj_func(individual) for individual in population])
        # cma_es.disp()
        
    fitness = cma_es.result.fbest
    
    return fitness