

import numpy as np
from scipy.integrate import solve_ivp

def heat_function(t, T, Q_func, Tamb, U, m, Cp, A, beta, eps, sigma):
    """
    Differential equation of the nonlinear energy balance.
    
    Parameters:
    -----------
    t : float
        Current time
    T : float
        Current temperature
    Q_func : callable
        Function that returns heat input at time t
    Tamb : float
        Ambient temperature
    U : float
        Heat transfer coefficient
    m : float
        Mass of the system
    Cp : float
        Heat capacity
    A : float
        Surface area
    beta : float
        Conversion factor for heat input
    eps : float
        Emissivity
    sigma : float
        Stefan-Boltzmann constant
    
    Returns:
    --------
    dTdt : float
        Time derivative of the temperature
    """
    Q = Q_func(t) # Evaluate heat input at time t
    dTdt = (beta * Q / (m * Cp)
            - U * A * (T - Tamb) / (m * Cp)
            - eps * sigma * A * (T**4 - Tamb**4) / (m * Cp))
    return dTdt

def heat_ode(T0, Tamb, U, m, Cp, A, beta, eps, sigma):
    """
    Solve the thermal system ODE using SciPy's solve_ivp with RK45.
    
    Parameters:
    -----------
    T0 : float
        Initial temperature
    Tamb : float or np.ndarray
        Ambient temperature (constant or variable)
    U, m, Cp, A, beta, eps, sigma : floats
        System parameters
    
    Returns:
    --------
    time : ndarray
        Time points where the solution is evaluated
    T : ndarray
        Temperature evolution over time
    Q_vals : ndarray
        Heat input values at each time point
    """
    n = 60 * 60  # Simulation length = 60 minutes (in seconds)
    time = np.linspace(1, n, n)  # Time vector for evaluation

    # Define Q as a function of time (in this case, rectangular pulse starting at the origin)
    width = 3 * n / 5.0 # Pulse width
    start_point = 0 * n / 5.0
    stop_point = start_point + 4 * width / 5
    end_point = start_point + width
    def Q_func(t):
        a = 100/(stop_point - start_point)
        b = -a * start_point
        if start_point <= t <= stop_point:
            return a*t + b
        elif stop_point < t <= end_point:
            return 200.0
        else:
            return 0.0

    # Solve the ODE using RK45 (adaptive Runge-Kutta)
    solve_edo = solve_ivp(fun = lambda t, T: heat_function(t, T, Q_func, Tamb, U, m, Cp, A, beta, eps, sigma),
                          t_span = (time[0], time[-1]), # Integration interval,
                          y0 = [T0],    # Initial condition
                          t_eval = time,    # Return solution at these times
                          method="RK45", dense_output=True)

    # Reconstruct Q values at the sampled times
    Q_vals = np.array([Q_func(t) for t in solve_edo.t])

    return solve_edo.t, solve_edo.y[0], Q_vals