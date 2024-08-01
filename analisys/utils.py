import numpy as np
import pandas as pd

def get_data(run_folder, speed='7_0', drop_columns=True, only_valids=True, top_ten=True):
    csv_file = '../results/' + run_folder + '/' + speed + '/all_results.csv'

    df = pd.read_csv(csv_file)

    # Remove colunas desnecessárias
    if drop_columns:
        df = df.drop(['Strength', 'Strength_Min', 'Cavitation', 'Cavitation_Max', 'Tip_Velocity', 'Tip_Velocity_Max'], axis=1)

    # Formata os dados
    df = df.astype({"P_B": float, "Valid": bool, "Run": int, "Generation": int})

    if only_valids:
        df = df[df['Valid'] == True]
    
    # Obtém as 10 melhores soluções
    if top_ten:
        df = df.sort_values(by='P_B')
        df = df.iloc[:10]
    
    return df