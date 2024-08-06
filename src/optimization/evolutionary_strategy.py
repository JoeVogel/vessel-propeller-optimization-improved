import random
import csv
import numpy as np
import os
import pandas as pd

import cma
from optimization.openai import OpenAIES
from optimization.hgso import HGSO
from optimization.pso import PSO

from vessel_propeller_evaluation.evaluate_propeller import evaluate
from multiprocessing.pool import ThreadPool 
from multiprocessing import Pool
from enum import Enum

class Solver(Enum):
    CMA_ES = 1
    OPENAI_ES = 2
    HGSO = 3
    PSO = 4

class EvolutionaryStrategy:
    
    def __init__(self, solver:Solver, population_size=255, max_generations=None, qtde_seeds=1, service_speed=7.0, b_series_json=None, number_of_blades=None, run_folder=None, verbose=False):
        
        # CMA_ES and OpenAI_ES
        self.sigma_init             = 0.1
        # OpenAI_ES              
        self.sigma_decay            = 0.999    
        self.learning_rate          = 0.01    
        self.learning_rate_decay    = 0.9999   
        self.weight_decay           = 0.01
        self.antithetic             = False   
        self.forget_best            = True
        # HGSO
        self.alpha          = 0.01
        self.beta           = 0.01
        self.K              = 0.5
        self.epxilon        = 0.05
        self.num_clusters   = 1
        # PSO
        self.c1     = 1.0
        self.c2     = 2.0 
        self.weight = 0.5
        # General
        self.generation_counter = 1
        self.run_number         = 0
        self.solver             = solver
        self.population_size    = population_size
        self.max_generations    = max_generations
        self.qtde_seeds         = qtde_seeds
        self.b_series           = b_series_json
        self.run_folder         = run_folder
        self.verbose            = verbose
        self.all_results        = pd.DataFrame(columns=['V_S',
                                                        'Z', 
                                                        'D', 
                                                        'AEdAO', 
                                                        'PdD', 
                                                        'P_B', 
                                                        'Strength', 
                                                        'Strength_Min', 
                                                        'Cavitation', 
                                                        'Cavitation_Max', 
                                                        'Tip_Velocity', 
                                                        'Tip_Velocity_Max', 
                                                        'Generation', 
                                                        'Run', 
                                                        'Valid'])

        if service_speed == None:
            raise Exception('Must provide service_speed parameter')
        
        self.service_speed      = service_speed
        
        if number_of_blades == None:
            raise Exception('Must provide number_of_blades parameter')
        
        self.number_of_blades = number_of_blades
        
    def get_valids(self):
        return self.all_results[self.all_results['Valid'] == True]

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
    
    def configure_hgso(self, alpha, beta, K, epxilon, num_clusters):
        self.alpha          = alpha
        self.beta           = beta
        self.K              = K
        self.epxilon        = epxilon
        self.num_clusters   = num_clusters

    def configure_pso(self, c1, c2, weight):
        self.c1     = c1
        self.c2     = c2
        self.weight = weight
     
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
            elif self.solver.value == 3:
                self.__run_hgso(lower_bounds=lower_bounds, upper_bounds=upper_bounds, seed=seed)
            elif self.solver.value == 4:
                self.__run_pso(lower_bounds=lower_bounds, upper_bounds=upper_bounds, seed=seed)
    
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
       
    def fitness_function(self, x):
        
        V_S     = self.service_speed
        D       = x[0]
        AEdAO   = x[1]
        PdD     = x[2]
        Z       = self.number_of_blades
        
        P_B, strength,strengthMin, cavitation,cavitationMax, velocity,velocityMax = evaluate(V_S,D,Z,AEdAO,PdD)

        cavitation_penalty = max(((cavitation - cavitationMax)/cavitationMax), 0) 
        tip_velocity_penalty = max(((velocity - velocityMax)/velocityMax), 0)
        strenght_penalty = max(((strengthMin - strength)/strengthMin), 0)

        penalty = cavitation_penalty + tip_velocity_penalty + strenght_penalty
                
        # Fitness is Power Brake multiplied by 1 + the percentage of each constraint
        fit_value = P_B * (1 + penalty)

        if self.verbose:
            print("Seed      ", self.run_number)
            print("Generation", self.generation_counter)
            print("V_S       ", V_S)
            print("Z         ", Z)
            print("D         ", x[0])
            print("AEdAO     ", x[1])
            print("PdD       ", x[2])
            print("Penalty", ("Yes" if penalty != 0 else "No"))
            print("Fitness", fit_value)
            print("")

        if (self.generation_counter > 0): # Se for 0 significa que é a geração da inicialização

            new_row = {
                'V_S':V_S,  
                'Z':Z,
                'D':D,
                'AEdAO':AEdAO,
                'PdD':PdD,
                'P_B':fit_value,
                'Strength':strength,
                'Strength_Min':strengthMin,
                'Cavitation':cavitation,
                'Cavitation_Max':cavitationMax,
                'Tip_Velocity':velocity,
                'Tip_Velocity_Max':velocityMax,
                'Generation':self.generation_counter,
                'Run':self.run_number,
                'Valid':(True if penalty == 0 else False)
            }

            if len(x) == 3:
                self.all_results.loc[len(self.all_results)] = new_row
            elif len(x) == 4: # HGSO altera aleatóriamente um dos valores da população, portanto precisa ser substituido
                filtered_df = self.all_results[(self.all_results['Generation'] == self.generation_counter) & (self.all_results['Run'] == self.run_number)]
                id = int(x[3])
                real_index = filtered_df.index[id]
                self.all_results.loc[real_index] = new_row
            else:
                raise Exception('Wrong number of parameters')
        
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
            self.es.tell(population, [self.fitness_function(individual) for individual in population])
            self.es.disp()
            
            self.generation_counter += 1
        
        self.es.result_pretty()
    
    def __run_openai_es(self, num_params=None, x0=None, lower_bounds=None, upper_bounds=None):
        
        self.es = OpenAIES(num_params,
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
        while (self.max_generations >= self.generation_counter):
            
            population = self.es.ask()
            
            fitness_list = np.zeros(self.es.popsize)
            
            for j in range(len(population)):
                fitness_list[j] = self.fitness_function(population[j])
                
            self.es.tell(fitness_list)
            
            result = self.es.result() # first element is the best solution, second element is the best fitness
            
            self.generation_counter += 1
    
    def __run_hgso(self, lower_bounds=None, upper_bounds=None, seed=None):
        
        self.generation_counter = 0 # Para desconsiderar a população inicial gerada e avaliada na inicialização
        
        obj_func = self.fitness_function
        
        hgso = HGSO(obj_func, 
                    lower_bounds, 
                    upper_bounds, 
                    alpha=self.alpha,
                    beta=self.beta,
                    epxilon=self.epxilon,
                    K=self.K,
                    verbose=True, 
                    pop_size=self.population_size,
                    n_clusters=self.num_clusters, 
                    random_seed=seed)

        self.generation_counter = 1

        while self.max_generations >= self.generation_counter:
            pos, fit, train_loss = hgso.solve(self.generation_counter)
            self.generation_counter += 1

    def __run_pso(self, lower_bounds=None, upper_bounds=None, seed=None):

        self.generation_counter = 0 # Para desconsiderar a população inicial gerada e avaliada na inicialização
        
        obj_func = self.fitness_function

        pso = PSO(obj_func=obj_func, 
                  dimension=3, 
                  lower_bounds=lower_bounds, 
                  upper_bounds=upper_bounds, 
                  num_particles=self.population_size, 
                  seed=seed)
        
        self.generation_counter = 1
        
        while self.max_generations >= self.generation_counter:
            pso.solve()
            self.generation_counter += 1
