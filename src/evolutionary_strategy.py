import random
import json
import numpy as np
import cma

from evaluate_propeller import evaluate
from multiprocessing.pool import ThreadPool 
from multiprocessing import Pool
from enum import Enum

class Solver(Enum):
    CMA_ES = 1
    OPENAI_ES = 2

class EvolutionaryStrategy:
    
    def __init__(self, solver:Solver, population_size=255, max_generations=None, qtde_seeds=1, service_speed=7.0, b_series_json=None):
        
        self.solver             = solver
        self.population_size    = population_size
        self.max_generations     = max_generations # run solver for this generations
        self.qtde_seeds         = qtde_seeds
        self.valid_solutions    = {
                                    "Z":[],
                                    "D":[],
                                    "AEdAO":[],
                                    "PdD":[],
                                    "P_B":[],
                                    "Strength":[],
                                    "Cavitation":[],
                                    "Tip Velocity":[],
                                    "Generation":[],
                                    "Run":[],} 
        self.service_speed      = service_speed
        
        if b_series_json == None:
            raise Exception('Must provide b_series_json parameter')
        
        self.b_series           = b_series_json
    
    def run_solver(self):
        
        lower_bounds = [self.b_series['range_D'][0], self.b_series['range_AEdAO'][0], self.b_series['range_PdD'][0]]
        upper_bounds = [self.b_series['range_D'][1], self.b_series['range_AEdAO'][1], self.b_series['range_PdD'][1]]
            
        for seed in range(1, self.qtde_seeds + 1):
        
            # random.seed(seed)
            
            x0 =  [
                    random.uniform(self.b_series['range_D'][0],     self.b_series['range_D'][1]),
                    random.uniform(self.b_series['range_AEdAO'][0], self.b_series['range_AEdAO'][1]),
                    random.uniform(self.b_series['range_PdD'][0],   self.b_series['range_PdD'][1])
                ]

            # print("D    ", x0[0])
            # print("AEdAO", x0[1])
            # print("PdD  ", x0[2])
            
            if self.solver.value == 1:
                self.__run_cma(x0=x0, lower_bounds=lower_bounds, upper_bounds=upper_bounds, run_number=seed)
        
    def __fitness_function(self, x, Z, generation, run_number):
        D     = x[0]
        AEdAO = x[1]
        PdD   = x[2]
        
        print("D    ", x[0])
        print("AEdAO", x[1])
        print("PdD  ", x[2])
        
        P_B, strength,strengthMin, cavitation,cavitationMax, velocity,velocityMax = evaluate(self.service_speed,D,Z,AEdAO,PdD)

        cavitation_penalty = max(((cavitation - cavitationMax)/cavitationMax), 0) 
        tip_velocity_penalty = max(((velocity - velocityMax)/velocityMax), 0)
        strenght_penalty = max(((strengthMin - strength)/strengthMin), 0)

        penalty = cavitation_penalty + tip_velocity_penalty + strenght_penalty
        
        # Fitness is Power Brake multiplied by 1 + the percentage of each constraint
        fit_value = P_B * (1 + penalty)
        
        if penalty == 0:            
            self.valid_solutions['Z'].append(Z)
            self.valid_solutions['D'].append(D)
            self.valid_solutions['AEdAO'].append(AEdAO)
            self.valid_solutions['PdD'].append(PdD)
            self.valid_solutions['P_B'].append(P_B)
            self.valid_solutions['Strength'].append(strength)
            self.valid_solutions['Cavitation'].append(cavitation)
            self.valid_solutions['Tip Velocity'].append(velocity)
            self.valid_solutions['Generation'].append(generation)
            self.valid_solutions['Run'].append(run_number)
            
        # we want the minimal P_B
        # the solvers use the max value as best fitness, so
        fit_value *= -1
        
        return fit_value   
        
    def __run_cma(self, num_params=None, x0=None, sigma_init=0.10, lower_bounds=None, upper_bounds=None, run_number=1):
        
        if x0 == None:
            if num_params == None:
                raise Exception("One of the parameters num_params or x0 must be provided!")
            
            x0 = np.zeros(num_params)

        #TODO: implement paralelism 
        for number_of_blades in range(self.b_series['range_Z'][0], self.b_series['range_Z'][1]):
        
            self.es = cma.CMAEvolutionStrategy(x0,
                                            sigma_init,
                                            {'popsize': self.population_size,
                                            'bounds': [lower_bounds, upper_bounds]})
            
            generation_counter = 1
            
            while not self.es.stop() or generation_counter <= self.max_generations:
                population = self.es.ask()
                self.es.tell(population, [self.__fitness_function(individual, number_of_blades, generation_counter, run_number) for individual in population])
                self.es.disp()
                generation_counter = generation_counter + 1
            
            self.es.result_pretty()
        
    # def get_best_result(self):
    #     # get result with best fitness
    #     best_result = max(self.solutions, key=(lambda x: x[2]))
    #     # print the values
    #     Z             =  best_result[0]
    #     D, AEdAO, PdD =  best_result[1]
    #     fitness       = -best_result[2]
    #     print("D:",D,"Z:",Z,"AEdAO:",AEdAO,"PdD:",PdD)
    #     print("fitness:",fitness)
    #     return best_result
    
    # def solver_for_Z(self, Z, V_S, seed):
    #     random.seed(seed)
    #     x0 =  [
    #             random.uniform(self.range_D[0],     self.range_D[1]),
    #             random.uniform(self.range_AEdAO[0], self.range_AEdAO[1]),
    #             random.uniform(self.range_PdD[0],   self.range_PdD[1])
    #         ]

    #     # defines CMA-ES algorithm solver
    #     cmaes = cma.CMAEvolutionStrategy(x0, 
    #                                         self.sigma, {'popsize': self.population_size, 
    #                                                     'bounds': [self.lower_bounds, self.upper_bounds]})

        
    #     # run the solver
    #     history, best_solution = self.test_solver(cmaes, V_S, Z)
        
    #     # print best solution
    #     D       = best_solution[0]
    #     AEdAO   = best_solution[1]
    #     PdD     = best_solution[2]
    #     fitness = history[-1]
    #     print("Z:",Z, "Best Solution:", best_solution, 'with fitness:', fitness)

    #     return [Z, best_solution, fitness, history] 
    
    
    