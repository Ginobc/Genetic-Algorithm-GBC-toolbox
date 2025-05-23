import numpy as np

def newpop(N_ind, CromLim, bounds_shape, config):
    if config['modo'] == 'continuous':      # Para contínuo, usa-se limites de variáveis reais
        Ncrom = CromLim.shape[0]
        pop = np.random.uniform(CromLim[:, 0], CromLim[:, 1], size=(N_ind, Ncrom))
        return CromLim, pop, None
    
    elif config['modo'] == 'discrete':      # Para discreto, considera-se índices discretos dos valores possíveis
        pop_idx = np.zeros((N_ind, bounds_shape), dtype=int)
        for j in range(bounds_shape):
            pop_idx[:, j] = np.random.randint(CromLim[j, 0], CromLim[j, 1] + 1, size=N_ind)
        return CromLim, None, pop_idx


def fitness(pop, fit_function, config, pop_idx):
    if config['modo'] == 'continuous':
        OUTPUT = np.array([fit_function(ind)[0] for ind in pop])
    elif config['modo'] == 'discrete':
        OUTPUT = np.array([fit_function(ind)[0] for ind in pop_idx])
    
    if config['modo_otimizacao'] == 'nsga2':
        fit = None  # dummy fitness for compatibility
    else:
        fit = 1 / (OUTPUT + 10)
    return OUTPUT, fit

def evolution_strategies(pop, fit_function, config, pop_idx, fit, p_elit, p_m, p_c, N_ind, CromLim, bounds_shape):
    if config['modo_otimizacao'] != 'nsga2':
        if config['modo'] == 'discrete':
            base = pop_idx
        else:
            base = pop
        N_ind, Ncrom = base.shape
        N_elit = int(np.floor(N_ind * p_elit))

        # Roulette-wheel Selection
        q = np.cumsum(fit/np.sum(fit))
        r = np.random.rand(N_ind)
        seleciona = np.searchsorted(q, r)
        sorted_idx = np.argsort(-fit)

        # Initialization
        new_base = np.zeros_like(base)
        new_base[:N_elit] = base[sorted_idx[:N_elit]]
        # new_pop = np.zeros_like(pop)

        # # Elitism
        # new_base[:N_elit] = base[sorted_idx[:N_elit]]
        # if config['modo'] == 'discrete':
        #     for j in range(bounds_shape):
        #         new_pop[:, j] = Crom[Names_CromLim[j]][new_base[:, j]]
        # else:
        #     new_pop[:N_elit, :] = new_base[:N_elit, :]

        # Evolution (Crossover & Mutation)
        i = N_elit
        while i < N_ind:
            if np.random.rand() >= p_m:
                pai = base[seleciona[np.random.randint(0, N_ind)]]
                mae = base[seleciona[np.random.randint(0, N_ind)]]
                if np.random.rand() <= p_c:
                    if config['crossover'] == "blx":
                        child = blx_alpha_crossover(pai, mae, CromLim, config['modo'])
                    elif config['crossover'] == "one_point":
                        child = one_point_crossover(pai, mae)
                    elif config['crossover'] == "two_point":
                        child = two_point_crossover(pai, mae)
                    else:
                        raise ValueError("Invalid crossover type. Choose 'blx', 'one_point', or 'two_point'.")
                else:
                    idx_pai = np.where((base == pai).all(axis=1))[0][0]
                    idx_mae = np.where((base == mae).all(axis=1))[0][0]
                    child = pai if fit[idx_pai] > fit[idx_mae] else mae

                # new_base[i] = child
                # if config['modo'] == 'discrete':
                #     for j in range(bounds_shape):
                #         new_pop[i, j] = Crom[Names_CromLim[j]][child[j]]
                # else:
                #     new_pop[i, :] = child
            else:
                if config['modo'] == 'discrete':
                    child = np.random.randint(CromLim[:, 0], CromLim[:, 1] + 1)
                else:
                    _, child, _ = newpop(1, CromLim, bounds_shape, config)
                    child = child[0]
            
            new_base[i] = child
            i += 1

        new_pop = None if config['modo'] == 'discrete' else new_base
        new_pop_idx = new_base if config['modo'] == 'discrete' else None

        return new_pop, new_pop_idx
    else:
        # Avaliar múltiplos objetivos
        obj1 = np.zeros(N_ind)
        obj2 = np.zeros(N_ind)
        for i, ind in enumerate(pop):
            f, _ = fit_function(ind)
            if isinstance(f, (int, float, np.number)):
                obj1[i] = f
                obj2[i] = f + 1.0
            else:
                obj1[i] = f[0]
                obj2[i] = f[1]

        # Ordenação não dominada
        domination_counts = np.zeros(N_ind)
        domination_sets = [[] for _ in range(N_ind)]
        ranks = np.zeros(N_ind)
        fronts = [[]]

        for p in range(N_ind):
            for q in range(N_ind):
                if ((obj1[p] <= obj1[q] and obj2[p] <= obj2[q]) and
                    (obj1[p] < obj1[q] or obj2[p] < obj2[q])):
                    domination_sets[p].append(q)
                elif ((obj1[q] <= obj1[p] and obj2[q] <= obj2[p]) and
                    (obj1[q] < obj1[p] or obj2[q] < obj2[p])):
                    domination_counts[p] += 1
            if domination_counts[p] == 0:
                ranks[p] = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in domination_sets[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        ranks[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        # Cálculo de crowding distance
        def crowding_distance(obj1, obj2, front):
            N = len(front)
            distance = np.zeros(N)
            for objs in [obj1, obj2]:
                sorted_idx = np.argsort([objs[i] for i in front])
                distance[sorted_idx[0]] = distance[sorted_idx[-1]] = float('inf')
                for i in range(1, N - 1):
                    prev_val = objs[front[sorted_idx[i - 1]]]
                    next_val = objs[front[sorted_idx[i + 1]]]
                    distance[sorted_idx[i]] += (next_val - prev_val) / (max(objs) - min(objs) + 1e-9)
            return distance

        # Seleção elitista baseada em dominância e distância
        new_pop = []
        for front in fronts[:-1]:
            if len(new_pop) + len(front) > N_ind:
                cd = crowding_distance(obj1, obj2, front)
                sorted_front = [front[i] for i in np.argsort(-cd)]
                new_pop.extend(sorted_front[:N_ind - len(new_pop)])
                break
            new_pop.extend(front)

        selected = pop[new_pop]

        # Cruzamento e mutação
        offspring = []
        while len(offspring) < N_ind:
            p1, p2 = selected[np.random.randint(len(selected), size=2)]
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2
            mutation = np.random.normal(0, 0.1, Ncrom)
            child += mutation
            child = np.clip(child, CromLim[:, 0], CromLim[:, 1])
            offspring.append(child)

        return np.array(offspring)

# Crossover strategies
def blx_alpha_crossover(parent1, parent2, CromLim, modo, alpha=0.25):
    beta = np.random.uniform(-alpha, 1 + alpha)

    if modo == 'discrete':
        child = np.round(parent1 + beta * (parent2 - parent1)).astype(int)
    elif modo == 'continuous':
        child = parent1 + beta * (parent2 - parent1)
        
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
