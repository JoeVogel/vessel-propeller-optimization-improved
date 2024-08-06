import json

from optimization.evolutionary_strategy import Solver

def load_b_series():
    pass

def load_hyperparameters(solver:Solver):
    
    file = None
    
    match solver.value:
        case 1:
            file = open('./data/cma_hyperparameters.json')
        case 2:
            file = open('./data/openai_hyperparameters.json')
        case 3:
            file = open('./data/hgso_hyperparameters.json')
        case 4:
            file = open('./data/pso_hyperparameters.json')
        case default:
            raise Exception('Invalid solver')
    
    data = json.load(file)
    
    return data