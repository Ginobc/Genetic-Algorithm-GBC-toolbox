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
