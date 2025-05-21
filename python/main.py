import numpy as np
import matplotlib.pyplot as plt
import time
from example import *
from ga_core import *

# Global configs
config = {
    'modo': 'continuous',              # 'continuous' or 'discrete'
    'modo_otimizacao': 'traditional',  # 'nsga2' or 'traditional'
    'exemple_name': 'easom',           # fitness function
    'crossover': 'blx',                # opções: 'sbx', 'blx', 'one_point', 'two_point'
}

# Mapping: example -> function
exemple = {
    'sphere': sphere_function,
    'easom': easom_function,
    'hadel': hadel_function,
    'simple': simple_function,
    'real_multi': real_multiobjective,
}
fit_function = exemple[config['exemple_name']]
_, CromLim = fit_function(None)
bounds_shape = CromLim.shape[0]

# Configuração do GA - Traditional
N_ger = 250
N_ind = 250
p_diz = 0.2
N_diz = 20
p_elit = 0.02
p_m = 0.02
p_c = 0.6

# Execução
melhor = []
medio = []
Crom = {}
Names_CromLim = []
start_time = time.time()

CromLim, pop, pop_idx = newpop(N_ind, CromLim, bounds_shape, Crom, Names_CromLim, config)

for i in range(N_ger):
    OUTPUT, fit = fitness(pop, fit_function, config, pop_idx, Crom, Names_CromLim, bounds_shape)
    pop, pop_idx = evolution_strategies(pop, fit_function, config, pop_idx, fit, p_elit, p_m, p_c, N_ind, Crom, CromLim, Names_CromLim, bounds_shape)

    if i % N_diz == 0:
        start_idx = int(N_ind * (1 - p_diz))
        if config['modo'] == 'continuous':
            _, pop[int(N_ind*(1-p_diz)):],_      = newpop(N_ind - start_idx, CromLim, bounds_shape, Crom, Names_CromLim, config)
        else:
            _, _, pop_idx[int(N_ind*(1-p_diz)):] = newpop(N_ind - start_idx, CromLim, bounds_shape, Crom, Names_CromLim, config)
        
    if config['modo_otimizacao'] == 'nsga2':
        obj1_gen = []
        for ind in pop:
            f, _ = fit_function(ind)
            if isinstance(f, (int, float, np.number)):
                obj1_gen.append(f)
            else:
                obj1_gen.append(f[0])  # usa apenas o primeiro objetivo para traçar
        melhor.append(np.min(obj1_gen))  # minimização
        medio.append(np.mean(obj1_gen))
    else:
        melhor.append(np.max(fit))
        medio.append(np.mean(fit))

if config['modo_otimizacao'] == 'nsga2':
    obj1 = []
    obj2 = []
    for ind in pop:
        f, _ = fit_function(ind)
        if isinstance(f, (int, float, np.number)):
            obj1.append(f)
            obj2.append(f + 1.0)  # objetivo fictício
        else:
            obj1.append(f[0])
            obj2.append(f[1])
    obj1 = np.array(obj1)
    obj2 = np.array(obj2)
    
    best_idx = np.argmin(obj1)
    best_x = pop[best_idx]
    best_val, _ = fit_function(best_x)  # valor real da função
    if isinstance(best_val, (list, np.ndarray)):
        print(f"Best objective values (f1, f2): {best_val[0]:.10f}, {best_val[1]:.10f}")
    else:
        print(f"Best objective value (f(x)): {best_val:.10f}")
    print(f"Best input (x): {np.round(best_x, 6)}")
    normalizador = np.max(np.abs(melhor))
    
    plt.figure()
    plt.scatter(obj1, obj2, label='Pareto Front')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title(f"NSGA-II ({config['exemple_name'].capitalize()} function)")
    plt.grid(True)
    plt.legend()

else:
    best_idx = np.argmax(fit)
    best_x = pop[best_idx]
    best_val, _ = fit_function(best_x)  # valor real da função
    print(f"Best objective value (f(x)): {best_val:.10f}")
    print(f"Best input (x): {np.round(best_x, 6)}")
    normalizador = np.max(melhor)

# Resultados
elapsed_time = time.time() - start_time
print(f"\nOptimization time: {elapsed_time:.2f} seconds")

plt.figure()
plt.plot(melhor / normalizador, label='Best')
plt.plot(medio / normalizador, label='Mean')
plt.legend()
plt.xlabel('Generations')
plt.ylabel('Fitness (Normalized)')
plt.title(f"Optimization ({config['modo'].capitalize()} - {config['exemple_name'].capitalize()} function)")
plt.grid(True)
plt.show()
