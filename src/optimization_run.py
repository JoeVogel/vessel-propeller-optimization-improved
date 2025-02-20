import json
import pandas as pd
import time
import os
import csv

from datetime import datetime
from optimization.evolutionary_strategy import Solver, EvolutionaryStrategy
from data_loader import load_hyperparameters

columns = ['V_S',
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
            'Valid']

def get_best(file_path):
    df = pd.read_csv(file_path) 
    
    df['Valid'] = df['Valid'].astype(bool)
    
    df_valid = df[df['Valid'] == True]
    
    has_valids = False
    
    if (len(df_valid.index) > 0):
        min_pb_row = df_valid.loc[df_valid['P_B'].idxmin()]
    
        has_valids = True
    
        return has_valids, min_pb_row
    
    return has_valids

def save_run_configs(run_folder, header, data):
    
    with open(run_folder + '/configs.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(header)
        writer.writerow(data)

def print_stats(run_folder):
    print()

    print("vS | P_B | Z | D | AEdAO | PdD")
    
    for v_S in range_V_S:
        v_S_folder = str(v_S).replace('.', '_')
        has_valids, min_pb_row = get_best(run_folder + '/' + v_S_folder + '/all_results.csv')
        if has_valids: print(str(v_S), "{:.3f}".format(min_pb_row.iloc[5]), min_pb_row.iloc[1], "{:.3f}".format(min_pb_row.iloc[2]), "{:.3f}".format(min_pb_row.iloc[3]), "{:.3f}".format(min_pb_row.iloc[4]))

def run_cma(generations, population_size, range_V_S, seeds, b_series): 
    solver      = Solver.CMA_ES
    
    configs = load_hyperparameters(solver)
    
    sigma_init = configs['sigma']
    
    if population_size is None:
        population_size = configs['num_population']
    
    datetime_now = datetime.now()
    run_folder = 'results/' + str(solver.name) + '-' + datetime_now.strftime("%m_%d-%H_%M")
    
    os.mkdir(run_folder)
    
    start_time = time.time()
    
    for V_S in range_V_S:

        all_results = pd.DataFrame(columns=columns)
        
        V_S_str = str(V_S).replace('.', '_')
        vS_folder = run_folder + '/' + V_S_str
        
        os.mkdir(vS_folder)
    
        # #TODO: implement paralelism 
        for Z in range(b_series['range_Z'][0], b_series['range_Z'][1] + 1):
        
            Z_str = str(Z)
            Z_folder = vS_folder + '/' + Z_str
        
            os.mkdir(Z_folder)
            
            es = EvolutionaryStrategy(solver, 
                                    max_generations=generations, 
                                    population_size=population_size, 
                                    service_speed=V_S,
                                    qtde_seeds=seeds, 
                                    b_series_json=b_series, 
                                    number_of_blades=Z,
                                    run_folder=vS_folder,
                                    verbose=True)
            
            es.configure_cma(sigma_init)
            
            es.run_solver()
            
            df = es.get_valids()
            
            df.to_csv(Z_folder + '/valids.csv', index=False)

            all_results = pd.concat([all_results, es.all_results], ignore_index=True)
    
        all_results.to_csv(vS_folder+'/all_results.csv', index=False)
    
    end_time = time.time()
    
    elapsed_time = (end_time - start_time) / 60
    # print(f"Tempo decorrido: {elapsed_time:.2f} minutos")
    print(f"Tempo decorrido: {int(elapsed_time)} minutos")
    
    header = ["range_V_S", "Solver", "Population_Size", "Max_Generations", "Sigma_Init", "Seeds", "Elapsed_Time"]
    data = [range_V_S, solver.name, population_size, generations, sigma_init, seeds, elapsed_time]
    save_run_configs(run_folder, header, data)
    
    print_stats(run_folder)

def run_openai_es(generations, population_size, range_V_S, seeds, b_series):
    
    solver      = Solver.OPENAI_ES
    
    configs = load_hyperparameters(solver)

    sigma_init = configs['sigma_init']
    sigma_decay = configs['sigma_decay']
    learning_rate = configs['learning_rate']
    learning_rate_decay = configs['learning_rate_decay']
    weight_decay = configs['weight_decay']
    
    if population_size is None:
        population_size = configs['num_population']
    
    datetime_now = datetime.now()
    run_folder = 'results/' + str(solver.name) + '-' + datetime_now.strftime("%m_%d-%H_%M")
    
    os.mkdir(run_folder)
    
    start_time = time.time()
    
    for V_S in range_V_S:

        all_results = pd.DataFrame(columns=columns)

        V_S_str = str(V_S).replace('.', '_')
        vS_folder = run_folder + '/' + V_S_str
        
        os.mkdir(vS_folder)
    
        # #TODO: implement paralelism 
        for Z in range(b_series['range_Z'][0], b_series['range_Z'][1] + 1):
            
            Z_str = str(Z)
            Z_folder = vS_folder + '/' + Z_str
        
            os.mkdir(Z_folder)
            
            es = EvolutionaryStrategy(solver, 
                                    max_generations=generations, 
                                    population_size=population_size, 
                                    service_speed=V_S,
                                    qtde_seeds=seeds, 
                                    b_series_json=b_series, 
                                    number_of_blades=Z,
                                    run_folder=vS_folder,
                                    verbose=True)
            
            es.configure_openai_es(sigma_init=sigma_init, 
                                   sigma_decay=sigma_decay, 
                                   learning_rate=learning_rate, 
                                   learning_rate_decay=learning_rate_decay, 
                                   weight_decay=weight_decay, 
                                   antithetic=False, 
                                   forget_best=False)
            
            es.run_solver()
            
            df = es.get_valids()
            
            df.to_csv(Z_folder + '/valids.csv', index=False)

            all_results = pd.concat([all_results, es.all_results], ignore_index=True)
    
        all_results.to_csv(vS_folder+'/all_results.csv', index=False)
    
    end_time = time.time()
    
    elapsed_time = (end_time - start_time) / 60
    # print(f"Tempo decorrido: {elapsed_time:.2f} minutos")
    print(f"Tempo decorrido: {int(elapsed_time)} minutos")
    
    header = ["range_V_S", "Solver", "Population_Size", "Max_Generations", "Sigma_Init", "Seeds", "Elapsed_Time"]
    data = [range_V_S, solver.name, population_size, generations, sigma_init, seeds, elapsed_time]
    save_run_configs(run_folder, header, data)
    
    print_stats(run_folder)

def run_hgso(generations, population_size, range_V_S, seeds, b_series):
    
    solver  = Solver.HGSO
    
    configs = load_hyperparameters(solver)

    alpha = configs['alpha']
    beta = configs['beta']
    epsilon = configs['epsilon']
    K = configs['K']
    num_clusters = configs['num_clusters']
    
    if population_size is None:
        population_size = configs['num_population']

    datetime_now = datetime.now()
    run_folder = 'results/' + str(solver.name) + '-' + datetime_now.strftime("%m_%d-%H_%M")
    
    os.mkdir(run_folder)
    
    start_time = time.time()

    for V_S in range_V_S:

        all_results = pd.DataFrame(columns=columns)

        V_S_str = str(V_S).replace('.', '_')
        vS_folder = run_folder + '/' + V_S_str
        
        os.mkdir(vS_folder)
    
        # #TODO: implement paralelism 
        for Z in range(b_series['range_Z'][0], b_series['range_Z'][1] + 1):
        
            Z_str = str(Z)
            Z_folder = vS_folder + '/' + Z_str
        
            os.mkdir(Z_folder)
            
            es = EvolutionaryStrategy(solver, 
                                    max_generations=generations, 
                                    population_size=population_size, 
                                    service_speed=V_S,
                                    qtde_seeds=seeds, 
                                    b_series_json=b_series, 
                                    number_of_blades=Z,
                                    run_folder=vS_folder,
                                    verbose=True)
            

            es.configure_hgso(alpha, 
                              beta, 
                              K, 
                              epsilon, 
                              num_clusters)
            
            es.run_solver()
            
            df = es.get_valids()
            
            df.to_csv(Z_folder + '/valids.csv', index=False)

            all_results = pd.concat([all_results, es.all_results], ignore_index=True)
    
        all_results.to_csv(vS_folder+'/all_results.csv', index=False)

    end_time = time.time()
    
    elapsed_time = (end_time - start_time) / 60
    # print(f"Tempo decorrido: {elapsed_time:.2f} minutos")
    print(f"Tempo decorrido: {int(elapsed_time)} minutos")
    
    header = ["range_V_S", "Solver", "Population_Size", "Max_Generations", "Seeds", "Elapsed_Time"]
    data = [range_V_S, solver.name, population_size, generations, seeds, elapsed_time]
    save_run_configs(run_folder, header, data)
    
    print_stats(run_folder)

def run_pso(generations, population_size, range_V_S, seeds, b_series):
                  
    solver  = Solver.PSO
    
    configs = load_hyperparameters(solver)
 
    c1      = configs['c1']
    c2      = configs['c2']
    weight  = configs['weight']
    
    if population_size is None:
        population_size = configs['population']

    datetime_now = datetime.now()
    run_folder = 'results/' + str(solver.name) + '-' + datetime_now.strftime("%m_%d-%H_%M")
    
    os.mkdir(run_folder)
    
    start_time = time.time()

    for V_S in range_V_S:
        all_results = pd.DataFrame(columns=columns)
        V_S_str = str(V_S).replace('.', '_')
        vS_folder = run_folder + '/' + V_S_str
        
        os.mkdir(vS_folder)
    
        # #TODO: implement paralelism 
        for Z in range(b_series['range_Z'][0], b_series['range_Z'][1] + 1):
        
            Z_str = str(Z)
            Z_folder = vS_folder + '/' + Z_str
        
            os.mkdir(Z_folder)
            
            es = EvolutionaryStrategy(solver, 
                                    max_generations=generations, 
                                    population_size=population_size, 
                                    service_speed=V_S,
                                    qtde_seeds=seeds, 
                                    b_series_json=b_series, 
                                    number_of_blades=Z,
                                    run_folder=vS_folder,
                                    verbose=True)
            
            es.configure_pso(c1=c1, c2=c2, weight=weight)
            
            es.run_solver()
            
            df = es.get_valids()
            
            df.to_csv(Z_folder + '/valids.csv', index=False)

            all_results = pd.concat([all_results, es.all_results], ignore_index=True)
    
        all_results.to_csv(vS_folder+'/all_results.csv', index=False)
    
    end_time = time.time()
    
    elapsed_time = (end_time - start_time) / 60
    # print(f"Tempo decorrido: {elapsed_time:.2f} minutos")
    print(f"Tempo decorrido: {int(elapsed_time)} minutos")
    
    header = ["range_V_S", "Solver", "Population_Size", "Max_Generations", "C1", "C2", "Weight", "Seeds", "Elapsed_Time"]
    data = [range_V_S, solver.name, population_size, generations, c1, c2, weight, seeds, elapsed_time]
    save_run_configs(run_folder, header, data)
    
    print_stats(run_folder)

   
if __name__ == "__main__":
    
    file = open('./data/b_series.json')
    b_series = json.load(file)
     
    range_V_S = [7.0, 7.5, 8.0, 8.5]
    # range_V_S = [7.0]
    population_size = None #Se None, usa o resultado do tunning de hiper parametros
    generations = 30
    seeds = 10
    
    run_cma(generations, population_size, range_V_S, seeds, b_series)
    
    run_openai_es(generations, population_size, range_V_S, seeds, b_series)

    run_hgso(generations, population_size, range_V_S, seeds, b_series)

    run_pso(generations, population_size, range_V_S, seeds, b_series)
    
