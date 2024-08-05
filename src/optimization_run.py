import json
import pandas as pd
import time
import os
import csv

from datetime import datetime
from evolutionary_strategy import Solver, EvolutionaryStrategy

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
    
    #   sigma   num_population
    #   0.1153  98

    sigma_init  = 0.1153
    
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

    # sigma_init sigma_decay learning_rate learning_rate_decay weight_decay num_population
    # 0.9645      0.8660        0.0242              0.6535       0.6936             73    
    
    sigma_init = 0.9645
    sigma_decay = 0.8660
    learning_rate = 0.0242
    learning_rate_decay = 0.6535
    weight_decay = 0.6936
    
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

    # alpha     beta    epsilon K       num_population  num_clusters
    # 0.8616    0.9845  0.9630  1.4175  59              3

    alpha = 0.8616
    beta = 0.9845
    epsilon = 0.9630
    K = 1.4175
    num_clusters = 3

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

    #   c1      c2      weight      population
    #   0.9986  2.3070  0.1229      58  
    c1      = 0.9986
    c2      = 2.3070
    weight  = 0.1229

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
     
    # range_V_S = [7.0, 7.5, 8.0, 8.5]
    range_V_S = [7.0, 8.5]
    population_size = 10
    generations = 30
    seeds = 1
    
    # population_size = 98
    # run_cma(generations, population_size, range_V_S, seeds, b_series)
    
    # population_size = 73
    # run_openai_es(generations, population_size, range_V_S, seeds, b_series)

    # population_size = 59
    # run_hgso(generations, population_size, range_V_S, seeds, b_series)

    population_size = 58
    run_pso(generations, population_size, range_V_S, seeds, b_series)
    
