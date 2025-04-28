import numpy as np
import matplotlib.pyplot as plt
from example import sphere_function, eason_function, hadel_function, simple_function
from ga_continuous import *
from ga_discrete import *

# Configurações gerais
modo = 'continuous'  # 'continuous' ou 'discrete'
func_name = 'sphere' # 'sphere', 'eason', 'hadel', 'simple'

# Selecionando a função
if func_name == 'sphere':
    fit_function = sphere_function
elif func_name == 'eason':
    fit_function = eason_function
elif func_name == 'hadel':
    fit_function = hadel_function
elif func_name == 'simple':
    fit_function = simple_function
else:
    raise ValueError("Invalid function name selected.")

# Configuração do GA
N_ger = 500
N_ind = 500
p_diz = 0.2
N_diz = 20
p_elit = 0.02
p_m = 0.02
p_c = 0.6

# Execução
melhor = []
medio = []

if modo == 'continuous':
    # Para contínuo, usa-se limites de variáveis reais
    _, bounds = fit_function(np.zeros(bounds_shape := 3 if func_name == 'sphere' else 2))
    pop = newpop_continuous(N_ind, bounds)
    for i in range(N_ger):
        OUTPUT, fit = fitness_continuous(pop, fit_function)
        pop = evolution_strategies_continuous(pop, fit, p_elit, p_m, p_c, bounds)
        if i % N_diz == 0:
            pop[int(N_ind*(1-p_diz)):] = newpop_continuous(int(N_ind*p_diz), bounds)
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

# Resultados
print(f'Best fitness: {max(melhor):.6f}')

plt.plot(melhor/np.max(melhor), label='Best')
plt.plot(medio/np.max(melhor), label='Mean')
plt.legend()
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.title(f"Optimization ({modo.capitalize()} - {func_name.capitalize()} function)")
plt.grid(True)
plt.show()
