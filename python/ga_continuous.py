
import numpy as np

def newpop_continuous(Nind, CromLim):
    Ncrom = CromLim.shape[0]
    return np.random.uniform(CromLim[:, 0], CromLim[:, 1], size=(Nind, Ncrom))

def fitness_continuous(pop, fit_function):
    OUTPUT = np.array([fit_function(ind)[0] for ind in pop])
    fit = 1 / (OUTPUT + 10)
    return OUTPUT, fit

def evolution_strategies_continuous(pop, fit, p_elit, p_m, p_c, CromLim):
    N_ind = len(fit)
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
