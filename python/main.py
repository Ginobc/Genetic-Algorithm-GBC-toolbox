import numpy as np
import matplotlib.pyplot as plt
import time
from example import sphere_function, eason_function, hadel_function, simple_function, real_multiobjective
from ga_continuous import *
from ga_discrete import *

# Configurações gerais
modo = 'continuous'                 # 'continuous' or 'discrete'
modo_otimizacao = 'traditional'     # 'nsga2' or 'traditional'
func_name = 'eason'                 # 'sphere', 'eason', 'hadel', 'simple', 'real_multi'

# Mapeamento nome -> função
funcoes_disponiveis = {
    'sphere': sphere_function,
    'eason': eason_function,
    'hadel': hadel_function,
    'simple': simple_function,
    'real_multi': real_multiobjective,
}
try:
    fit_function = funcoes_disponiveis[func_name]
except KeyError:
    raise ValueError(f"Invalid function name selected: '{func_name}'")

# Configuração do GA - Traditional
N_ger = 200
N_ind = 200
p_diz = 0.2
N_diz = 20
p_elit = 0.02
p_m = 0.02
p_c = 0.6

# # Configuração do GA - Pareto
# N_ger = 1000
# N_ind = 500
# p_diz = 0.2
# N_diz = 20
# p_elit = 0.05
# p_m = 0.03
# p_c = 0.6

# Execução
melhor = []
medio = []

start_time = time.time()
if modo == 'continuous':
    # Para contínuo, usa-se limites de variáveis reais
    _, bounds = fit_function(np.zeros(bounds_shape := 3 if func_name == 'sphere' else 2))
    pop = newpop_continuous(N_ind, bounds)
    for i in range(N_ger):
        OUTPUT, fit = fitness_continuous(pop, fit_function, modo_otimizacao)
        pop = evolution_strategies_continuous(pop, fit_function, fit, p_elit, p_m, p_c, bounds, modo_otimizacao)
        if i % N_diz == 0:
            pop[int(N_ind*(1-p_diz)):] = newpop_continuous(int(N_ind*p_diz), bounds)
        if modo_otimizacao == 'nsga2':
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


elif modo == 'discrete':
    # Para discreto, considera-se índices discretos dos valores possíveis
    _, bounds = fit_function(np.zeros(bounds_shape := 3 if func_name == 'sphere' else 2))
    # Criar valores discretizados manualmente
    Crom = {}
    Names_CromLim = []
    for i in range(bounds_shape):
        var_name = f"x{i+1}"
        Names_CromLim.append(var_name)
        Crom[var_name] = np.linspace(bounds[i,0], bounds[i,1], 10)  # 10 níveis discretos entre limites

    CromLim, pop, pop_idx = newpop_discrete(N_ind, Crom, Names_CromLim)

    for i in range(N_ger):
        # Avaliação da fitness
        fit = np.zeros(N_ind)
        for j in range(N_ind):
            x_j = [Crom[Names_CromLim[k]][pop_idx[j, k]] for k in range(bounds_shape)]
            fit[j] = 1 / (fit_function(x_j)[0] + 10)

        pop_idx = evolution_strategies_discrete(pop_idx, fit, p_elit, p_m, p_c, N_ind, Crom, CromLim, Names_CromLim)

        for j in range(bounds_shape):
            pop[:, j] = Crom[Names_CromLim[j]][pop_idx[:, j]]

        if i % N_diz == 0:
            _, _, pop_idx[int(N_ind*(1-p_diz)):] = newpop_discrete(N_ind - int(N_ind*(1-p_diz)), Crom, Names_CromLim)

        melhor.append(np.max(fit))
        medio.append(np.mean(fit))

else:
    raise ValueError("Invalid mode. Choose 'continuous' or 'discrete'.")

if modo_otimizacao == 'nsga2':
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
    
# Resultados
elapsed_time = time.time() - start_time
print(f"\nOptimization time: {elapsed_time:.2f} seconds")

if modo_otimizacao == 'nsga2':
    best_idx = np.argmin(obj1)
    best_x = pop[best_idx]
    best_val, _ = fit_function(best_x)  # valor real da função
    if isinstance(best_val, (list, np.ndarray)):
        print(f"Best objective values (f1, f2): {best_val[0]:.10f}, {best_val[1]:.10f}")
    else:
        print(f"Best objective value (f(x)): {best_val:.10f}")
    print(f"Best input (x): {np.round(best_x, 6)}")
    normalizador = np.max(np.abs(melhor))

else:
    best_idx = np.argmax(fit)
    best_x = pop[best_idx]
    best_val, _ = fit_function(best_x)  # valor real da função
    print(f"Best objective value (f(x)): {best_val:.10f}")
    print(f"Best input (x): {np.round(best_x, 6)}")
    normalizador = np.max(melhor)

plt.figure()
plt.plot(melhor / normalizador, label='Best')
plt.plot(medio / normalizador, label='Mean')
plt.legend()
plt.xlabel('Generations')
plt.ylabel('Fitness (Normalized)')
plt.title(f"Optimization ({modo.capitalize()} - {func_name.capitalize()} function)")
plt.grid(True)

if modo_otimizacao == 'nsga2':
    plt.figure()
    plt.scatter(obj1, obj2, label='Pareto Front')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title(f"NSGA-II ({func_name.capitalize()} function)")
    plt.grid(True)
    plt.legend()

plt.show()
