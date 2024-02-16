import json

# from cmaes import CMAES
from evolutionary_strategy import Solver, EvolutionaryStrategy


if __name__ == "__main__":
    
    file = open('./data/b_series.json')
    b_series = json.load(file)
    
    es = EvolutionaryStrategy(Solver.CMA_ES, max_generations=30, population_size=30, b_series_json=b_series)
    es.run_solver()