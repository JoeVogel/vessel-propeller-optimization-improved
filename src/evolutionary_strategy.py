import random
import csv
import numpy as np
import os

import cma
from openai_es import OpenES

from evaluate_propeller import evaluate
from multiprocessing.pool import ThreadPool 
from multiprocessing import Pool
from enum import Enum

class Solver(Enum):
    CMA_ES = 1
    OPENAI_ES = 2

class EvolutionaryStrategy:
    
    def __init__(self, solver:Solver, population_size=255, max_generations=None, qtde_seeds=1, service_speed=7.0, b_series_json=None, number_of_blades=None, run_folder=None):
        
        # CMA_ES and OpenAI_ES
        self.sigma_init             = 0.1
        # OpenAI_ES              
        self.sigma_decay            = 0.999    
        self.learning_rate          = 0.01    
        self.learning_rate_decay    = 0.9999   
        self.weight_decay           = 0.01
        self.antithetic             = False   
        self.forget_best            = True
        # General
        self.generation_counter = 1
        self.run_number         = 0
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
        
        if run_folder == None:
            raise Exception('Must provide run folder parameter')
        
        self.run_folder = run_folder
    
    def configure_cma(self, sigma_init):
        self.sigma_init = sigma_init
     
    def configure_openai_es(self, sigma_init, sigma_decay, learning_rate, learning_rate_decay, weight_decay, antithetic, forget_best):
        self.sigma_init             = sigma_init   
        self.sigma_decay            = sigma_decay
        self.learning_rate          = learning_rate
        self.learning_rate_decay    = learning_rate_decay
        self.weight_decay           = weight_decay 
        self.antithetic             = antithetic   
        self.forget_best            = forget_best
     
    def run_solver(self):
        
        bounds          = [self.b_series['range_D'], self.b_series['range_AEdAO'], self.b_series['range_PdD']]
        lower_bounds    = [self.b_series['range_D'][0], self.b_series['range_AEdAO'][0], self.b_series['range_PdD'][0]]
        upper_bounds    = [self.b_series['range_D'][1], self.b_series['range_AEdAO'][1], self.b_series['range_PdD'][1]]
            
        for seed in range(0, self.qtde_seeds):
        
            # Seta seed do random para que a geração de números aleatórios seja reproduzível
            # Usado em x0 e também nas eurísticas dos algoritmos como o DE e o CMA 
            random.seed(seed)
            
            x0 =  [
                    random.uniform(self.b_series['range_D'][0],     self.b_series['range_D'][1]),
                    random.uniform(self.b_series['range_AEdAO'][0], self.b_series['range_AEdAO'][1]),
                    random.uniform(self.b_series['range_PdD'][0],   self.b_series['range_PdD'][1])
                ]
            
            self.run_number         = seed
            self.generation_counter = 1
            
            if self.solver.value == 1:
                self.__run_cma(x0=x0, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
            elif self.solver.value == 2:
                self.__run_openai_es(num_params=3, x0=x0, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    
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
       
    def __fitness_function(self, x):
        
        V_S     = self.service_speed
        D       = x[0]
        AEdAO   = x[1]
        PdD     = x[2]
        Z       = self.number_of_blades
        
        print("Seed      ", self.run_number)
        print("Generation", self.generation_counter)
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
        print("Penalty", ("Yes" if penalty != 0 else "No"))
        
        # Fitness is Power Brake multiplied by 1 + the percentage of each constraint
        fit_value = P_B * (1 + penalty)
        
        print("Fitness", fit_value)
        print("")
        
        csv_path = self.run_folder + '/all_results.csv'
        
        header = ["V_S", "Z", "D", "AEdAO", "PdD", "P_B", "Strength", "Strength_Min", "Cavitation", "Cavitation_Max", "Tip_Velocity", "Tip_Velocity_Max", "Generation", "Run", "Valid"]
        data = [V_S, Z, D, AEdAO, PdD, P_B, strength,strengthMin, cavitation,cavitationMax, velocity,velocityMax, self.generation_counter, self.run_number, (True if penalty == 0 else False)]
        
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            if not file_exists:
                writer.writerow(header)
            
            writer.writerow(data)
        
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
            self.valid_solutions['Generation'].append(self.generation_counter)
            self.valid_solutions['Run'].append(self.run_number)
            
        # we want the minimal P_B
        # the solvers use the max value as best fitness, so
        # fit_value *= -1
        
        return fit_value   
 
    def __run_cma(self, num_params=None, x0=None, lower_bounds=None, upper_bounds=None):
        
        if x0 == None:
            if num_params == None:
                raise Exception("One of the parameters num_params or x0 must be provided!")
            
            x0 = np.zeros(num_params)
       
        self.es = cma.CMAEvolutionStrategy(x0,
                                        self.sigma_init,
                                        {'popsize': self.population_size,
                                        'bounds': [lower_bounds, upper_bounds]})
        
        while not self.es.stop():
            
            if self.max_generations != None and self.generation_counter > self.max_generations:
                print('Maximum generations reached.')
                break
            
            population = self.es.ask()
            self.es.tell(population, [self.__fitness_function(individual, self.generation_counter) for individual in population])
            self.es.disp()
            
            self.generation_counter += 1
        
        self.es.result_pretty()
    
    def __run_openai_es(self, num_params=None, x0=None, lower_bounds=None, upper_bounds=None):
        
        self.es = OpenES(num_params,
                         x0=x0,
                         popsize=self.population_size,
                         sigma_init=self.sigma_init,
                         sigma_decay=self.sigma_decay,
                         learning_rate=self.learning_rate,
                         learning_rate_decay=self.learning_rate_decay,
                         weight_decay=self.weight_decay, 
                         antithetic=self.antithetic,
                         forget_best=self.forget_best,
                         lower_bounds=lower_bounds,
                         upper_bounds=upper_bounds )
        
        #TODO: implementar avaliação de convergencia
        for i in range(self.max_generations):
            
            population = self.es.ask()
            
            fitness_list = np.zeros(self.es.popsize)
            
            for j in range(len(population)):
                fitness_list[j] = self.__fitness_function(population[j])
                
            self.es.tell(fitness_list)
            
            result = self.es.result() # first element is the best solution, second element is the best fitness
            
            self.generation_counter += 1