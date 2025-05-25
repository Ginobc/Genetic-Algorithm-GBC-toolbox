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
    # Mapping of Crossover Operators
    crossover_methods = {
        'blx': lambda p1, p2: blx_alpha_crossover(p1, p2, CromLim, config['modo']),
        'sbx': lambda p1, p2: sbx_crossover(p1, p2, CromLim),
        'linear': linear_convex_crossover,
        'one_point': one_point_crossover,
        'two_point': two_point_crossover
    }  

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

        # Elitism
        new_base[:N_elit] = base[sorted_idx[:N_elit]]

        # Evolution (Crossover & Mutation) - traditional
        i = N_elit
        while i < N_ind:
            if np.random.rand() >= p_m:
                parent1 = base[seleciona[np.random.randint(0, N_ind)]]
                parent2 = base[seleciona[np.random.randint(0, N_ind)]]

                if np.random.rand() <= p_c:
                    # Crossover
                    child = crossover_methods[config['crossover']](parent1, parent2)
                else:
                    idx_p1 = np.where((base == parent1).all(axis=1))[0][0]
                    idx_p2 = np.where((base == parent2).all(axis=1))[0][0]
                    child = parent1 if fit[idx_p1] > fit[idx_p2] else parent2
            else:
                if config['modo'] == 'discrete':
                    child = np.random.randint(CromLim[:, 0], CromLim[:, 1] + 1)
                else:
                    _, new_sample, _ = newpop(1, CromLim, bounds_shape, config)
                    child = new_sample[0]

            new_base[i] = child
            i += 1

        new_pop = None if config['modo'] == 'discrete' else new_base
        new_pop_idx = new_base if config['modo'] == 'discrete' else None

        return new_pop, new_pop_idx
    else:
        # === Avaliação multiobjetivo (genérico para n ≥ 2) ===
        obj_vals = []
        for ind in pop:
            f, _ = fit_function(ind)
            obj_vals.append(f if isinstance(f, (list, np.ndarray)) else [f])
        objs = np.array(obj_vals)
        n_objs = objs.shape[1]
        N_ind = objs.shape[0]

        # === Ordenação não dominada (generalizada) ===
        domination_counts = np.zeros(N_ind, dtype=int)
        domination_sets = [[] for _ in range(N_ind)]
        ranks = np.full(N_ind, np.inf)
        fronts = [[]]

        for p in range(N_ind):
            for q in range(N_ind):
                if p == q:
                    continue
                if np.all(objs[p] <= objs[q]) and np.any(objs[p] < objs[q]):
                    domination_sets[p].append(q)
                elif np.all(objs[q] <= objs[p]) and np.any(objs[q] < objs[p]):
                    domination_counts[p] += 1
            if domination_counts[p] == 0:
                ranks[p] = 0
                fronts[0].append(p)

        # Construção dos próximos fronts
        i = 0
        while i < len(fronts) and len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in domination_sets[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        ranks[q] = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)

        # === Cálculo de crowding distance (n objetivos) ===
        def crowding_distance_multiobj(objs, front):
            Nf = len(front)
            distance = np.zeros(Nf)
            if Nf == 0:
                return distance
            for m in range(objs.shape[1]):
                obj_m = objs[:, m]
                sorted_idx = np.argsort([obj_m[i] for i in front])
                distance[sorted_idx[0]] = distance[sorted_idx[-1]] = float('inf')
                obj_min = np.min(obj_m)
                obj_max = np.max(obj_m)
                if obj_max - obj_min == 0:
                    continue
                for i in range(1, Nf - 1):
                    prev_val = obj_m[front[sorted_idx[i - 1]]]
                    next_val = obj_m[front[sorted_idx[i + 1]]]
                    distance[sorted_idx[i]] += (next_val - prev_val) / (obj_max - obj_min)
        
        # === Seleção elitista baseada em dominância e distância ===
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) > N_ind:
                cd = crowding_distance_multiobj(objs, front)
                sorted_front = [front[i] for i in np.argsort(-cd)]
                new_pop.extend(sorted_front[:N_ind - len(new_pop)])
                break
            new_pop.extend(front)

        selected = pop[new_pop]
        Ncrom = pop.shape[1]

        # === Crossover and Mutation (NSGA-II) ===
        offspring = []
        while len(offspring) < N_ind:
            parent1, parent2 = selected[np.random.randint(len(selected), size=2)]
            
            # Crossover
            child = crossover_methods[config['crossover']](parent1, parent2)

            # Mutation
            if config['modo'] == 'continuous':
                mutation = np.random.normal(0, 0.1, Ncrom)
                child += mutation
                child = np.clip(child, CromLim[:, 0], CromLim[:, 1])

            offspring.append(child)

        return np.array(offspring), None
    
# Crossover strategies
def sbx_crossover(parent1, parent2, CromLim, eta=15):
    """
    Simulated Binary Crossover (SBX) for continuous variables.
    Adaptado do artigo de Deb et al. (2002), ideal para NSGA-II.

    Parâmetros:
    - parent1, parent2: Vetores dos pais (numpy array)
    - CromLim: Limites das variáveis (Nx2)
    - eta: Parâmetro de distribuição. Quanto maior, mais próximos dos pais
    
    Mecanismo:
    - A distribuição de probabilidade do filho depende do quão perto ele está dos pais.
    - O parâ metro η (eta) controla isso:
        - η pequeno (ex: 2–5): filhos mais diversificados
        - η grande (ex: 20–100): filhos mais próximos dos pais

    Retorna:
    - Um único filho (numpy array)
    """
    child = np.empty_like(parent1)
    for i in range(len(parent1)):
        x1, x2 = parent1[i], parent2[i]
        xl, xu = CromLim[i, 0], CromLim[i, 1]

        if np.random.rand() <= 0.5:
            if abs(x1 - x2) > 1e-14:
                if x1 > x2:
                    x1, x2 = x2, x1
                rand = np.random.rand()
                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

                c = c1 if np.random.rand() < 0.5 else c2
                c = np.clip(c, xl, xu)
            else:
                c = x1
        else:
            c = x1

        child[i] = c
    return child

def blx_alpha_crossover(parent1, parent2, CromLim, modo, alpha=0.25):
    beta = np.random.uniform(-alpha, 1 + alpha)

    if modo == 'discrete':
        child = np.round(parent1 + beta * (parent2 - parent1)).astype(int)
    elif modo == 'continuous':
        child = parent1 + beta * (parent2 - parent1)
        
    for k in range(CromLim.shape[0]):
        child[k] = np.clip(child[k], CromLim[k, 0], CromLim[k, 1])

    return child

def linear_convex_crossover(parent1, parent2):
    alpha = np.random.rand()
    child = alpha * parent1 + (1 - alpha) * parent2
    return child

def one_point_crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1))
    child = np.concatenate((parent1[:point], parent2[point:]))
    return child

def two_point_crossover(parent1, parent2):
    point1, point2 = sorted(np.random.randint(1, len(parent1), 2))
    child = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
    return child
