import numpy as np

def newpop_continuous(Nind, CromLim):
    Ncrom = CromLim.shape[0]
    return np.random.uniform(CromLim[:, 0], CromLim[:, 1], size=(Nind, Ncrom))

def fitness_continuous(pop, fit_function, modo_otimizacao):
    if modo_otimizacao == 'nsga2':
        OUTPUT = np.array([fit_function(ind)[0] for ind in pop])
        fit = None  # dummy fitness for compatibility
    else:
        OUTPUT = np.array([fit_function(ind)[0] for ind in pop])
        fit = 1 / (OUTPUT + 10)
    return OUTPUT, fit

def evolution_strategies_continuous(pop, fit_function, fit, p_elit, p_m, p_c, CromLim, modo_otimizacao):
    N_ind = pop.shape[0]    
    Ncrom = pop.shape[1]

    if modo_otimizacao == 'nsga2':
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

    else:
        # Modo tradicional
        N_elit = int(np.round(N_ind * p_elit))
        new_pop = np.zeros_like(pop)

        P = fit/np.sum(fit)
        q = np.cumsum(P)
        r = np.random.rand(N_ind)
        seleciona = np.searchsorted(q, r)

        sorted_idx = np.argsort(-fit)
        new_pop[:N_elit, :] = pop[sorted_idx[:N_elit], :]

        i = N_elit
        while i < N_ind:
            if np.random.rand() >= p_m:
                pai = pop[seleciona[np.random.randint(0, N_ind)]]
                mae = pop[seleciona[np.random.randint(0, N_ind)]]
                if np.random.rand() <= p_c:
                    alpha = 0.25
                    beta = np.random.uniform(-alpha, 1+alpha)
                    child = pai + beta*(mae-pai)
                    child = np.clip(child, CromLim[:,0], CromLim[:,1])
                    new_pop[i, :] = child
                else:
                    new_pop[i, :] = pai if fit[np.where((pop == pai).all(axis=1))[0][0]] > fit[np.where((pop == mae).all(axis=1))[0][0]] else mae
            else:
                new_pop[i,:] = newpop_continuous(1, CromLim)
            i +=1
    return new_pop
