import json
import pandas as pd

from datetime import datetime
from evolutionary_strategy import Solver, EvolutionaryStrategy

all_valid_solutions = {
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

def append_results(run_results):
    all_valid_solutions["V_S"].extend(run_results["V_S"])
    all_valid_solutions["Z"].extend(run_results["Z"])
    all_valid_solutions["D"].extend(run_results["D"])
    all_valid_solutions["AEdAO"].extend(run_results["AEdAO"])
    all_valid_solutions["PdD"].extend(run_results["PdD"])
    all_valid_solutions["P_B"].extend(run_results["P_B"])
    all_valid_solutions["Strength"].extend(run_results["Strength"])
    all_valid_solutions["Strength_Min"].extend(run_results["Strength_Min"])
    all_valid_solutions["Cavitation"].extend(run_results["Cavitation"])
    all_valid_solutions["Cavitation_Max"].extend(run_results["Cavitation_Max"])
    all_valid_solutions["Tip_Velocity"].extend(run_results["Tip_Velocity"])
    all_valid_solutions["Tip_Velocity_Max"].extend(run_results["Tip_Velocity_Max"])
    all_valid_solutions["Generation"].extend(run_results["Generation"])
    all_valid_solutions["Run"].extend(run_results["Run"])

if __name__ == "__main__":
    
    file = open('./data/b_series.json')
    b_series = json.load(file)
     
    range_V_S = [7.0, 7.5, 8.0, 8.5]
    solver = Solver.CMA_ES
    
    for V_S in range_V_S:
    
        # #TODO: implement paralelism 
        for number_of_blades in range(b_series['range_Z'][0], b_series['range_Z'][1] + 1):
        
            es = EvolutionaryStrategy(solver, 
                                    max_generations=30, 
                                    population_size=5, 
                                    service_speed=V_S,
                                    qtde_seeds=10, 
                                    b_series_json=b_series, 
                                    number_of_blades=number_of_blades)
            es.run_solver()
            
            append_results(es.valid_solutions)

    df = pd.DataFrame(all_valid_solutions)
    datetime_now = datetime.now()
    file_name = 'results/' + str(solver.name) + '-' + datetime_now.strftime("%m_%d-%H_%M") + '.csv'     
    df.to_csv(file_name, index=False)
    
    df_7 = df[df["V_S"] == 7.0]
    best_solution_7 = df_7.loc[df_7['P_B'].idxmin()]
    
    df_7_5 = df[df["V_S"] == 7.5]
    best_solution_7_5 = df_7_5.loc[df_7_5['P_B'].idxmin()]
    
    df_8 = df[df["V_S"] == 8.0]
    best_solution_8 = df_8.loc[df_8['P_B'].idxmin()]
    
    df_8_5 = df[df["V_S"] == 8.5]
    best_solution_8_5 = df_8_5.loc[df_8_5['P_B'].idxmin()]
    
    print("Best 7  :", best_solution_7['P_B'])
    print("Best 7.5:", best_solution_7_5['P_B'])
    print("Best 8  :", best_solution_8['P_B'])
    print("Best 8.5:", best_solution_8_5['P_B'])