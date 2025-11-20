from scipy.integrate import solve_ivp
import numpy as np


def fhn_ode_surrogate(t, y, params):
    u, v, x, y_rot = y
    a, b, epsilon, k1, k2, omega, gamma = params
    
    du_dt = u * (1 - u) * (u - a) - v
    dv_dt = epsilon * (u - b * v)
    dx_dt = k1 * u - k2 * x 
    dy_dt = omega * x - gamma * y_rot 
    
    return [du_dt, dv_dt, dx_dt, dy_dt]
