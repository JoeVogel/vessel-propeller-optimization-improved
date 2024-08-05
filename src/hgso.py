   
import numpy as np
from numpy.random import uniform, seed
from copy import deepcopy

# Referências: 
#     https://www.geeksforgeeks.org/henry-gas-solubility-optimization/
#     https://www.geeksforgeeks.org/implementation-of-henry-gas-solubility-optimization/

"""
Parâmetros recomendados para ajuste:

self.K

    Motivo: Afeta a magnitude das variações de posição das partículas, 
    influenciando diretamente a exploração do espaço de busca. 
    Pode ser ajustado para melhorar a convergência.

self.alpha

    Motivo: Controla a influência das interações sociais 𝑆𝑖𝑗. 
    Ajustar este valor pode ajudar a equilibrar a exploração e a exploração, 
    dependendo da natureza do problema de otimização.

self.beta

    Motivo: Controla a influência da melhor solução local. 
    Ajustar este parâmetro pode melhorar a capacidade do algoritmo de refinar as 
    soluções encontradas.

self.epxilon

    Motivo: Embora geralmente pequeno, ajustar ligeiramente este valor pode 
    ajudar a evitar problemas numéricos específicos do seu problema de otimização.

self.l1, self.l2 e self.l3

    Motivo: Esses parâmetros definem os intervalos iniciais para 𝐻𝑗, 𝑃𝑖𝑗 e 𝐶𝑗, respectivamente. 
    Ajustar esses limites pode ser necessário para adequar o algoritmo ao seu 
    problema específico e melhorar o desempenho.

Estratégia geral para ajuste de parâmetros:

    Ajuste incremental: Alterar um parâmetro de cada vez e observar o impacto no 
    desempenho do algoritmo.
    
    Teste e validação: Use um conjunto de validação para testar diferentes configurações 
    de parâmetros e identificar quais ajustes proporcionam a melhor performance.

    Metaparametrização: Considere usar métodos automatizados, como a busca em grade ou 
    a otimização Bayesiana, para encontrar os melhores valores de parâmetros.

Exemplo de ajuste incremental:

    Comece ajustando 𝛼 e 𝛽 para ver como as alterações afetam a exploração.
    Ajuste 𝐾 para refinar a magnitude das variações de posição das partículas.
    Experimente diferentes valores para 𝑙1, 𝑙2 e 𝑙3 para adaptar os limites iniciais 
    ao seu problema específico.
"""


class HGSO():

    ID_MIN_PROB = 0 # min problem
    ID_MAX_PROB = -1 # max problem
    ID_POS = 0 # Position
    ID_FIT = 1 # Fitness

    def __init__(self, obj_func=None, lb=None, ub=None, 
                 alpha = 0.01, beta=0.01, epxilon = 0.05, K = 0.5, verbose=True, pop_size=100, 
                 n_clusters=1, random_seed=None, **kwargs):
        self.pop_size = pop_size
        self.n_clusters = n_clusters
        self.n_elements = int(self.pop_size / self.n_clusters)
        self.lb = lb
        self.ub = ub
        self.verbose = verbose
        self.T0 = 298.15 # Representa a temperatura inicial em Kelvin. Usada no cálculo de 𝐻𝑗 para simular um processo de resfriamento. Uma temperatura inicial alta pode ajudar a explorar mais o espaço de busca inicialmente, enquanto uma temperatura baixa pode ajudar na exploração mais local nos estágios finais do algoritmo.
        self.K = K  # Constante de Boltzmann. Utilizada no cálculo de 𝑆𝑖𝑗, que mede a energia de interação entre as partículas. Afeta a magnitude das variações de posição das partículas.
        self.alpha = alpha
        self.beta = beta
        self.epxilon = epxilon
        self.obj_func = obj_func
        self.l1 = 0.05 # Limite para o fator de energia de interação inicial.
        self.l2 = 1 # Limite para o fator de energia de potencial inicial.
        self.l3 = 0.01 # Limite para o fator de decaimento da energia de interação inicial.
        self.H_j = self.l1 * uniform()
        self.P_ij = self.l2 * uniform()
        self.C_j = self.l3 * uniform()
        self.solution, self.loss_train = None, []
        if random_seed is not None:
            seed(random_seed)
        
        self.pop, self.group = self.create_population__(self.ID_MIN_PROB, self.n_clusters)
        self.g_best = self.get_global_best_solution(self.pop, self.ID_FIT, self.ID_MIN_PROB)
        self.p_best = self.get_best_solution_in_team(self.group)
        
    def get_fitness_position(self, position=None, minmax=0):
        return self.obj_func(position) if minmax == 0 else 1.0 / (self.obj_func(position) + 10E-10)

    def get_fitness_solution(self, solution=None, minmax=0):
        return self.get_fitness_position(solution[self.ID_POS], minmax)

    def get_global_best_solution(self, pop=None, id_fit=None, id_best=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[id_fit])
        return deepcopy(sorted_pop[id_best])

    def update_global_best_solution(self, pop=None, id_best=None, g_best=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        current_best = sorted_pop[id_best]
        return deepcopy(current_best) if current_best[self.ID_FIT] < g_best[self.ID_FIT] else deepcopy(g_best)

    def create_population__(self, minmax=0, n_clusters=0):
        pop = []
        group = []
        for i in range(n_clusters):
            team = []
            for j in range(self.n_elements):
                solution = uniform(self.lb, self.ub)
                fitness = self.obj_func(solution) if minmax == 0 else 1.0 / (self.obj_func(solution) + 10E-10)
                team.append([solution, fitness, i])
                pop.append([solution, fitness, i])
            group.append(team)
        return pop, group

    def get_best_solution_in_team(self, group=None):
        list_best = []
        for i in range(len(group)):
            sorted_team = sorted(group[i], key=lambda temp: temp[self.ID_FIT])
            list_best.append(deepcopy(sorted_team[self.ID_MIN_PROB]))
        return list_best

    def solve(self, epoch):
        
        for i in range(self.n_clusters):
            for j in range(self.n_elements):
                F = -1.0 if uniform() < 0.5 else 1.0
                self.H_j = self.H_j * np.exp(-self.C_j * (1.0 / np.exp(-epoch / epoch) - 1.0 / self.T0))
                S_ij = self.K * self.H_j * self.P_ij
                gamma = self.beta * np.exp(- ((self.p_best[i][self.ID_FIT] + self.epxilon) / (self.group[i][j][self.ID_FIT] + self.epxilon)))
                X_ij = self.group[i][j][self.ID_POS] + F * uniform() * gamma * (self.p_best[i][self.ID_POS] - self.group[i][j][self.ID_POS]) + F * uniform() * self.alpha * (S_ij * self.g_best[self.ID_POS] - self.group[i][j][self.ID_POS])

                # Garantir que X_ij esteja dentro dos limites
                X_ij = np.clip(X_ij, self.lb, self.ub)

                fit = self.get_fitness_position(X_ij, self.ID_MIN_PROB)
                self.group[i][j] = [X_ij, fit, i]
                self.pop[i*self.n_elements + j] = [X_ij, fit, i]

        self.H_j = self.H_j * np.exp(-self.C_j * (1.0 / np.exp(-epoch / epoch) - 1.0 / self.T0))
        S_ij = self.K * self.H_j * self.P_ij
        N_w = int(self.pop_size * (uniform(0, 0.1) + 0.1))
        sorted_id_pos = np.argsort([x[self.ID_FIT] for x in self.pop])

        for item in range(N_w):
            id = sorted_id_pos[item]
            j = id % self.n_elements
            i = int((id-j) / self.n_elements)
            X_new = uniform(self.lb, self.ub)
            X_new_flagged = np.append(X_new, id) # Adiciona o id ao final para que este elemento substitua o individuo anterior na lista da fitness_function
            fit = self.get_fitness_position(X_new_flagged, self.ID_MIN_PROB)
            self.pop[id] = [X_new, fit, i]
            self.group[i][j] = [X_new, fit, i]

        self.p_best = self.get_best_solution_in_team(self.group)
        self.g_best = self.update_global_best_solution(self.pop, self.ID_MIN_PROB, self.g_best)
        self.loss_train.append(self.g_best[self.ID_FIT])

        if self.verbose:
            print("Epoch: {}, Best fitness value: {}".format(epoch + 1, self.g_best[self.ID_FIT]))
        
        self.solution = self.g_best
        return self.g_best[self.ID_POS], self.g_best[self.ID_FIT], self.loss_train