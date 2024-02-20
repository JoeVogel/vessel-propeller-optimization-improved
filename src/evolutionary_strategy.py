import random
import csv
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
    
    def __init__(self, solver:Solver, population_size=255, max_generations=None, qtde_seeds=1, service_speed=7.0, b_series_json=None, number_of_blades=None):
        
        self.solver             = solver
        self.population_size    = population_size
        self.max_generations    = max_generations
        self.qtde_seeds         = qtde_seeds
        self.valid_solutions    = {
                                    "V_S":[],
                                    "Z":[],
                                    "D":[],
                                    "AEdAO":[],
                                    "PdD":[],
                                    "P_B":[],
                                    "Strength":[],
                                    "Strength_Min":[],
                                    "Cavitation":[],
                                    "Cavitation_Max":[],
                                    "Tip_Velocity":[],
                                    "Tip_Velocity_Max":[],
                                    "Generation":[],
                                    "Run":[]
                                } 
        
        if service_speed == None:
            raise Exception('Must provide service_speed parameter')
        
        self.service_speed      = service_speed
        
        if b_series_json == None:
            raise Exception('Must provide b_series_json parameter')
        
        self.b_series           = b_series_json
        
        if number_of_blades == None:
            raise Exception('Must provide number_of_blades parameter')
        
        self.number_of_blades = number_of_blades
    
    def run_solver(self):
        
        lower_bounds = [self.b_series['range_D'][0], self.b_series['range_AEdAO'][0], self.b_series['range_PdD'][0]]
        upper_bounds = [self.b_series['range_D'][1], self.b_series['range_AEdAO'][1], self.b_series['range_PdD'][1]]
            
        for seed in range(1, self.qtde_seeds + 1):
        
            random.seed(seed)
            
            x0 =  [
                    random.uniform(self.b_series['range_D'][0],     self.b_series['range_D'][1]),
                    random.uniform(self.b_series['range_AEdAO'][0], self.b_series['range_AEdAO'][1]),
                    random.uniform(self.b_series['range_PdD'][0],   self.b_series['range_PdD'][1])
                ]
            
            if self.solver.value == 1:
                self.__run_cma(x0=x0, sigma_init=2.0, lower_bounds=lower_bounds, upper_bounds=upper_bounds, run_number=seed)
        
    def __fitness_function(self, x, generation, run_number):
        V_S     = self.service_speed
        D       = x[0]
        AEdAO   = x[1]
        PdD     = x[2]
        Z       = self.number_of_blades
        
        print("Seed      ", run_number)
        print("Generation", generation)
        print("V_S       ", V_S)
        print("Z         ", Z)
        print("D         ", x[0])
        print("AEdAO     ", x[1])
        print("PdD       ", x[2])
        
        P_B, strength,strengthMin, cavitation,cavitationMax, velocity,velocityMax = evaluate(V_S,D,Z,AEdAO,PdD)

        cavitation_penalty = max(((cavitation - cavitationMax)/cavitationMax), 0) 
        tip_velocity_penalty = max(((velocity - velocityMax)/velocityMax), 0)
        strenght_penalty = max(((strengthMin - strength)/strengthMin), 0)

        penalty = cavitation_penalty + tip_velocity_penalty + strenght_penalty
        
        # Fitness is Power Brake multiplied by 1 + the percentage of each constraint
        fit_value = P_B * (1 + penalty)
        
        print("Fitness", fit_value)
        print("Penalty", ("Yes" if penalty != 0 else "No"))
        print("")
        
        if penalty == 0:  
            self.valid_solutions['V_S'].append(V_S)  
            self.valid_solutions['Z'].append(Z)
            self.valid_solutions['D'].append(D)
            self.valid_solutions['AEdAO'].append(AEdAO)
            self.valid_solutions['PdD'].append(PdD)
            self.valid_solutions['P_B'].append(fit_value)
            self.valid_solutions['Strength'].append(strength)
            self.valid_solutions['Strength_Min'].append(strengthMin)
            self.valid_solutions['Cavitation'].append(cavitation)
            self.valid_solutions['Cavitation_Max'].append(cavitationMax)
            self.valid_solutions['Tip_Velocity'].append(velocity)
            self.valid_solutions['Tip_Velocity_Max'].append(velocityMax)
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
       
        self.es = cma.CMAEvolutionStrategy(x0,
                                        sigma_init,
                                        {'popsize': self.population_size,
                                        'bounds': [lower_bounds, upper_bounds]})
        
        generation_counter = 1
        
        while not self.es.stop():
            
            if self.max_generations != None and generation_counter > self.max_generations:
                print('Maximum generations reached.')
                break
            
            population = self.es.ask()
            self.es.tell(population, [self.__fitness_function(individual, generation_counter, run_number) for individual in population])
            self.es.disp()
            
            generation_counter = generation_counter + 1
        
        self.es.result_pretty()
        
    def get_best_result(self):
        # get result with best fitness 
        
        if len(self.valid_solutions['P_B']) == 0:
            return None
         
        min_P_B = min(self.valid_solutions['P_B'])
        
        index_of_best = self.valid_solutions['P_B'].index(min_P_B)
        
        # print the values
        Z       = self.valid_solutions['Z'][index_of_best]
        D       = self.valid_solutions['D'][index_of_best]
        AEdAO   = self.valid_solutions['AEdAO'][index_of_best]
        PdD     = self.valid_solutions['PdD'][index_of_best]
        fitness = self.valid_solutions['P_B'][index_of_best]
        
        # print("D:",D,"Z:",Z,"AEdAO:",AEdAO,"PdD:",PdD)
        # print("fitness:",fitness)
        return [Z, D, AEdAO, PdD, fitness]
    