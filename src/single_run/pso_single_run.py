import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from optimization.pso import PSO
from optimization.evolutionary_strategy import EvolutionaryStrategy, Solver

def run(instance, w, c1, c2, num_particles):

    lower_bounds    = [0.5, 0.3, 0.5]
    upper_bounds    = [0.8, 1.05, 1.4]
    
    es = EvolutionaryStrategy(Solver.PSO, service_speed=8.5, number_of_blades=6)
    obj_func = es.fitness_function
    
    pso = PSO(obj_func=obj_func, dimension=3, lower_bounds=lower_bounds, upper_bounds=upper_bounds, num_particles=num_particles, w=w, c1=c1, c2=c2, seed=0)

    max_iter = 30
    for _ in range(max_iter):
        pso.solve()

    solution, fitness = pso.get_result()
    # print(fitness)
    return fitness