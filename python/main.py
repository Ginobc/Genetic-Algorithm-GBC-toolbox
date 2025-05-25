import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import time
from example import *
from ga_core import *

# Global configs
config = {
    # modo: 'continuous' or 'discrete'
    'modo': 'continuous',   
    # modo_otimização: 'nsga2' (only continuous) or 'traditional'                  
    'modo_otimizacao': 'nsga2',       
    # example_aname: sphere, easom, hadel, simple, real_multi_2v (NSGA2 w 2 objs), real_multi_3v (NSGA2 w 3 objs), discrete_alloy (discrete example)
    'exemple_name': 'real_multi_3v',    
    # crossover: 'sbx' (only continuous), 'blx', 'linear', 'one_point', 'two_point'           
    'crossover': 'sbx',                     
}

# Mapping: example -> function
example = {
    'sphere': sphere_function,
    'easom': easom_function,
    'hadel': hadel_function,
    'simple': simple_function,
    'real_multi_2v': real_multiobjective,       # 2 objectives - example_name = real_multi & modo_otimizacao = nsga2
    'real_multi_3v': real_multiobjective_2,     # 3 objectives - example_name = real_multi & modo_otimizacao = nsga2
    'discrete_alloy': discrete_alloy_optimization   # example_name = discrete_alloy
}
fit_function = example[config['exemple_name']]
_, CromLim = fit_function(None)
bounds_shape = CromLim.shape[0]

# Configuração do GA
N_ger = 100
N_ind = 100
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

CromLim, pop, pop_idx = newpop(N_ind, CromLim, bounds_shape, config)

for i in range(N_ger):
    OUTPUT, fit = fitness(pop, fit_function, config, pop_idx)
    pop, pop_idx = evolution_strategies(pop, fit_function, config, pop_idx, fit, p_elit, p_m, p_c, N_ind, CromLim, bounds_shape)

    if i % N_diz == 0:
        start_idx = int(N_ind * (1 - p_diz))
        if config['modo'] == 'continuous':
            _, pop[int(N_ind*(1-p_diz)):],_      = newpop(N_ind - start_idx, CromLim, bounds_shape, config)
        else:
            _, _, pop_idx[int(N_ind*(1-p_diz)):] = newpop(N_ind - start_idx, CromLim, bounds_shape, config)
        
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
    import plotly.express as px

    # Avaliação multiobjetivo
    objs = []
    for ind in pop:
        f, _ = fit_function(ind)
        objs.append(f if isinstance(f, (list, np.ndarray)) else [f])
    objs = np.array(objs)
    n_objs = objs.shape[1]

    # Criar DataFrame com variáveis e objetivos
    df_pareto = pd.DataFrame(pop, columns=[f'x{i}' for i in range(pop.shape[1])])
    for i in range(n_objs):
        df_pareto[f'f{i+1}'] = objs[:, i]

    # Identificação de mínimos
    idx_min_f1 = np.argmin(objs[:, 0])
    idx_min_f2 = np.argmin(objs[:, 1])
    if n_objs >= 3:
        idx_min_f3 = np.argmin(objs[:, 2])

    # Verificação de dominância
    def is_dominated(sol, others):
        return np.any(np.all(others <= sol, axis=1) & np.any(others < sol, axis=1))

    obj_array = df_pareto[[f'f{i+1}' for i in range(n_objs)]].values
    mask_nd = np.array([not is_dominated(obj_array[i], np.delete(obj_array, i, axis=0))
                        for i in range(len(obj_array))])
    df_pareto['Dominated'] = ~mask_nd

    # Cálculo do ponto balanceado (entre os não dominados)
    norm_objs = (objs - objs.min(axis=0)) / (objs.max(axis=0) - objs.min(axis=0) + 1e-8)
    dist_to_origin = np.linalg.norm(norm_objs, axis=1)
    idx_balanced = np.argmin(dist_to_origin[mask_nd])
    idx_balanced = np.where(mask_nd)[0][idx_balanced]

    # Marcar tipos
    df_pareto['type'] = ''
    df_pareto.loc[idx_min_f1, 'type'] = 'min_f1'
    df_pareto.loc[idx_min_f2, 'type'] = 'min_f2'
    if n_objs >= 3:
        df_pareto.loc[idx_min_f3, 'type'] = 'min_f3'
    df_pareto.loc[idx_balanced, 'type'] = 'balanced'

    # Gerar DataFrame apenas com as não dominadas (com os tipos marcados)
    df_nd = df_pareto[~df_pareto['Dominated']].copy()

    # Exportar
    df_pareto.to_excel("pareto_solutions.xlsx", index=False)
    df_nd.to_excel("pareto_front_non_dominated.xlsx", index=False)

    # Impressão dos representativos
    print("\nSummary of representative Pareto-optimal solutions:")
    labels = ['min_f1', 'min_f2', 'balanced']
    indices = [idx_min_f1, idx_min_f2, idx_balanced]
    if n_objs >= 3:
        labels.append('min_f3')
        indices.append(idx_min_f3)

    for label, idx in zip(labels, indices):
        f_sol = objs[idx]
        x_sol = pop[idx]
        obj_str = ', '.join([f"f{i+1} = {val:.4f}" for i, val in enumerate(f_sol)])
        print(f"{label.upper():>9}: {obj_str} | x = {np.round(x_sol, 4)}")

    # Gráficos
    if n_objs == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(df_nd['f1'], df_nd['f2'], c='blue', s=20, label='Pareto front')

        for tipo, cor in zip(['min_f1', 'min_f2', 'balanced'], ['red', 'green', 'orange']):
            p = df_nd[df_nd['type'] == tipo]
            if not p.empty:
                plt.scatter(p['f1'], p['f2'], c=cor, s=80, edgecolors='black', label=tipo.capitalize())
                x_cols = [col for col in df_nd.columns if col.startswith('x')]
                x_vals = p.iloc[0][x_cols].values
                txt = f"{[round(float(val), 2) for val in np.atleast_1d(x_vals)]}"
                plt.annotate(txt, (p['f1'].values[0], p['f2'].values[0]), fontsize=8)

        plt.xlabel("Objective 1 (f1)")
        plt.ylabel("Objective 2 (f2)")
        plt.title(f"NSGA-II ({config['exemple_name'].capitalize()} function)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    elif n_objs >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # f1 vs f2
        axes[0].scatter(df_nd['f1'], df_nd['f2'], c='blue', s=20, label='Pareto front')
        for tipo, cor in zip(['min_f1', 'min_f2', 'balanced'], ['red', 'green', 'orange']):
            p = df_nd[df_nd['type'] == tipo]
            if not p.empty:
                axes[0].scatter(p['f1'], p['f2'], c=cor, s=80, edgecolors='black', label=tipo.capitalize())
        axes[0].set_xlabel("f1")
        axes[0].set_ylabel("f2")
        axes[0].set_title("Pareto front: f1 vs f2")
        axes[0].legend()
        axes[0].grid(True)

        # f1 vs f3
        axes[1].scatter(df_nd['f1'], df_nd['f3'], c='blue', s=20, label='Pareto front')
        for tipo, cor in zip(['min_f1', 'min_f3', 'balanced'], ['red', 'purple', 'orange']):
            p = df_nd[df_nd['type'] == tipo]
            if not p.empty:
                axes[1].scatter(p['f1'], p['f3'], c=cor, s=80, edgecolors='black', label=tipo.capitalize())
        axes[1].set_xlabel("f1")
        axes[1].set_ylabel("f3")
        axes[1].set_title("Pareto front: f1 vs f3")
        axes[1].legend()
        axes[1].grid(True)

        plt.suptitle(f"NSGA-II ({config['exemple_name'].capitalize()} function)")
        plt.tight_layout()
        plt.show()

        # Gráfico 3D interativo
        fig = px.scatter_3d(df_nd,
                            x='f1', y='f2', z='f3',
                            color='type',
                            symbol='type',
                            opacity=0.9,
                            title="3D Interactive Pareto Front (Non-dominated)",
                            labels={'f1': 'Objective 1', 'f2': 'Objective 2', 'f3': 'Objective 3'})
        fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')))
        fig.update_layout(
            legend_title_text='Solution Type',
            scene=dict(
                aspectmode='cube',
                xaxis_title='Objective 1',
                yaxis_title='Objective 2',
                zaxis_title='Objective 3',
            )
        )
        try:
            fig.show()
        except:
            fig.write_html("pareto_3d_plot.html")
            print("Plot 3D salvo como 'pareto_3d_plot.html'")

    # Normalizador
    normalizador = np.max(np.abs(melhor))

else:
    best_idx = np.argmax(fit)
    if config['modo'] == 'discrete':
        best_x = pop_idx[best_idx]
    else:
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
