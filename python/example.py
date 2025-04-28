import numpy as np

def sphere_function(x):
    y = x[0]**2 + x[1]**2 + x[2]**2
    bounds = np.array([[-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12]])
    return y, bounds

def eason_function(x):
    y = -np.cos(x[0])*np.cos(x[1])*np.exp(-((x[0]-np.pi)**2 + (x[1]-np.pi)**2))
    bounds = np.array([[-50, 50], [-50, 50]])
    return y, bounds

def hadel_function(x):
    y = x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1]) + 0.7
    bounds = np.array([[-2, 2], [-2, 2]])
    return y, bounds

def simple_function(x):
    y = 2*x[0]**2 + 2*x[0]*x[1] + 2*x[1]**2 - 6*x[0]
    bounds = np.array([[-2, 2], [-2, 2]])
    return y, bounds
