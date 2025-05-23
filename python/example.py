import numpy as np

def sphere_function(x):
    bounds = np.array([[-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12]])
    if x is None:
        return None, bounds
    y = x[0]**2 + x[1]**2 + x[2]**2
    return y, bounds

def easom_function(x):
    bounds = np.array([[-50, 50], [-50, 50]])
    if x is None:
        return None, bounds
    y = -np.cos(x[0])*np.cos(x[1])*np.exp(-((x[0]-np.pi)**2 + (x[1]-np.pi)**2))
    return y, bounds

def hadel_function(x):
    bounds = np.array([[-2, 2], [-2, 2]])
    if x is None:
        return None, bounds
    y = x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1]) + 0.7
    return y, bounds

def simple_function(x):
    bounds = np.array([[-2, 2], [-2, 2]])
    if x is None:
        return None, bounds
    y = 2*x[0]**2 + 2*x[0]*x[1] + 2*x[1]**2 - 6*x[0]
    return y, bounds

def real_multiobjective(x):
    bounds = np.array([[-2, 4], [-3, 3]])  # faixa que cobre os dois pontos ótimos
    if x is None:
        return None, bounds
    f1 = x[0]**2 + x[1]**2
    f2 = (x[0] - 2)**2 + (x[1] + 1)**2
    return [f1, f2], bounds

def discrete_alloy_optimization(x):
    """
    Otimização discreta de composição de liga metálica com 3 metais.
    Cada variável representa o percentual de um metal, com valores pré-definidos.
    """
    Crom = {
        "metal_A": np.array([10, 20, 30, 40, 50]),
        "metal_B": np.array([10, 20, 30, 40, 50]),
        "metal_C": np.array([10, 20, 30, 40, 50])
    }
    names = list(Crom.keys())

    # Retorna bounds apenas na chamada inicial do main.py
    if x is None:
        bounds = np.array([[0, len(Crom[k]) - 1] for k in names])
        return None, bounds

    percentuais = [Crom[names[i]][x[i]] for i in range(len(x))]

    total = sum(percentuais)
    percentuais = [p / total for p in percentuais]

    custos = [5.0, 3.0, 2.0]        # $/kg
    resistencias = [300, 200, 100] # MPa
    densidades = [7.8, 8.5, 6.0]   # g/cm³

    custo_total = sum(p * c for p, c in zip(percentuais, custos))
    resistencia_total = sum(p * r for p, r in zip(percentuais, resistencias))
    densidade_total = sum(p * d for p, d in zip(percentuais, densidades))

    penalidade = (densidade_total - 8.0) ** 2 if densidade_total > 8.0 else 0
    objetivo = custo_total - 0.01 * resistencia_total + 100 * penalidade

    return objetivo, None


