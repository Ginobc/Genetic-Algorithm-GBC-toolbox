import numpy as np

# Escolha o tipo de crossover: 'blx', 'one_point', 'two_point'
crossover_type = "one_point"

def blx_alpha_crossover(parent1, parent2, CromLim, alpha=0.25):
    a2 = -alpha
    b2 = 1 + alpha
    beta = a2 + (b2 - a2) * np.random.rand()
    child = np.round(parent1 + beta * (parent2 - parent1)).astype(int)
    
    for k in range(CromLim.shape[0]):
        child[k] = np.clip(child[k], CromLim[k, 0], CromLim[k, 1])
    
    return child

def one_point_crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1))
    child = np.concatenate((parent1[:point], parent2[point:]))
    return child

def two_point_crossover(parent1, parent2):
    point1, point2 = sorted(np.random.randint(1, len(parent1), 2))
    child = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
    return child

def newpop_discrete(N_ind, Crom, Names_CromLim):
    pop_idx = np.zeros((N_ind, len(Names_CromLim)), dtype=int)
    pop = np.zeros((N_ind, len(Names_CromLim)))
    CromLim = np.zeros((len(Names_CromLim), 2))

    for j, name in enumerate(Names_CromLim):
        CromLim[j] = [0, len(Crom[name]) - 1]
        pop_idx[:, j] = np.random.randint(0, len(Crom[name]), size=N_ind)
        pop[:, j] = Crom[name][pop_idx[:, j]]

    return CromLim, pop, pop_idx

def evolution_strategies_discrete(pop_idx, fit, p_elit, p_m, p_c, N_ind, Crom, CromLim, Names_CromLim):
    q = np.cumsum(fit/np.sum(fit))
    r = np.random.rand(N_ind)
    seleciona = np.searchsorted(q, r)
    N_elit = int(np.floor(N_ind * p_elit))
    new_pop = np.zeros_like(pop_idx)

    sorted_idx = np.argsort(-fit)
    new_pop[:N_elit] = pop_idx[sorted_idx[:N_elit]]

    i = N_elit
    while i < N_ind:
        if np.random.rand() >= p_m:
            pai = pop_idx[seleciona[np.random.randint(0, N_ind)]]
            mae = pop_idx[seleciona[np.random.randint(0, N_ind)]]
            if np.random.rand() <= p_c:
                if crossover_type == "blx":
                    child = blx_alpha_crossover(pai, mae, CromLim)
                elif crossover_type == "one_point":
                    child = one_point_crossover(pai, mae)
                elif crossover_type == "two_point":
                    child = two_point_crossover(pai, mae)
                else:
                    raise ValueError("Invalid crossover type. Choose 'blx', 'one_point' or 'two_point'.")
                new_pop[i] = child
            else:
                new_pop[i] = pai if fit[np.where((pop_idx == pai).all(axis=1))[0][0]] > fit[np.where((pop_idx == mae).all(axis=1))[0][0]] else mae
        else:
            _, _, new_pop[i] = newpop_discrete(1, Crom, Names_CromLim)
        i += 1

    return new_pop
