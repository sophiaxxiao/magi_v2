import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

##############################
# Define the ODE system
##############################
def f_vec(t, X, thetas):
    '''
    Log-scale EIR model with E implicitly represented.
    ODE:
      dE/dt = beta * S * I - sigma * E
      dI/dt = sigma * E - gamma * I
      dR/dt = gamma * I
    where S = 1 - E - I - R

    On the log scale: X = [logE, logI, logR]

    Parameters
    ----------
    t : float
        Time
    X : np.array (3,)
        [logE, logI, logR]
    thetas : array-like
        [beta, gamma, sigma]

    Returns
    -------
    dXdt : np.array (3,)
        Derivatives in log scale
    '''
    logE, logI, logR = X
    beta, gamma, sigma = thetas

    E = np.exp(logE)
    I = np.exp(logI)
    R = np.exp(logR)

    S = 1.0 - E - I - R
    if S <= 0.0:
        S = 1e-10  # small positive value to avoid instability

    # Original scale derivatives
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

    # Log scale derivatives
    dlogEdt = dEdt / E
    dlogIdt = dIdt / I
    dlogRdt = dRdt / R

    return np.array([dlogEdt, dlogIdt, dlogRdt])

def ODE_log_scale(t, y, theta):
    return f_vec(t, y, theta)

##############################
# Negative log-likelihood function
##############################
# Negative log-likelihood function
def negative_log_likelihood(params, ts_obs, logI_obs, logR_obs):
    log_beta, log_gamma, log_sigma, logE0, logI0, logR0 = params

    beta = np.exp(log_beta)
    gamma = np.exp(log_gamma)
    sigma = np.exp(log_sigma)

    thetas = [beta, gamma, sigma]
    X0 = np.array([logE0, logI0, logR0])

    # Solve ODE
    sol = solve_ivp(lambda t, y: ODE_log_scale(t, y, thetas),
                    t_span=(ts_obs[0], ts_obs[-1]),
                    y0=X0, t_eval=ts_obs, rtol=1e-10, atol=1e-10)

    if not sol.success:
        return np.inf

    X_pred = sol.y.T
    logI_pred = X_pred[:, 1]
    logR_pred = X_pred[:, 2]

    mask_I = ~np.isnan(logI_obs)
    mask_R = ~np.isnan(logR_obs)

    residuals_I = (logI_obs[mask_I] - logI_pred[mask_I])**2
    residuals_R = (logR_obs[mask_R] - logR_pred[mask_R])**2

    # Assume known measurement noise variance on log scale
    sigma_obs = 0.1
    var_obs = sigma_obs**2

    sse = np.sum(residuals_I) + np.sum(residuals_R)
    nll = sse / (2 * var_obs)

    return nll

# Main function for running optimization and visualization
def mle(ts_obs, X_obs_full, maxiter=1000):
    logI_obs = X_obs_full[:, 1]
    logR_obs = X_obs_full[:, 2]

    init_log_beta = 0.5 * np.random.randn()
    init_log_gamma = 0.5 * np.random.randn()
    init_log_sigma = 0.5 * np.random.randn()
    init_logI0 = logI_obs[0] if not np.isnan(logI_obs[0]) else -4.0
    init_logR0 = logR_obs[0] if not np.isnan(logR_obs[0]) else -4.0
    init_logE0 = -4.0

    init_guess = [init_log_beta, init_log_gamma, init_log_sigma, init_logE0, init_logI0, init_logR0]
    bounds = [(-5, 5), (-3, 3), (-3, 3), (-10, 0), (-10, 5), (-10, 5)]

    res = minimize(negative_log_likelihood, init_guess, args=(ts_obs, logI_obs, logR_obs),
                   method='L-BFGS-B', bounds=bounds, options={'maxiter': maxiter})

    log_beta_est, log_gamma_est, log_sigma_est, logE0_est, logI0_est, logR0_est = res.x
    beta_est = np.exp(log_beta_est)
    gamma_est = np.exp(log_gamma_est)
    sigma_est = np.exp(log_sigma_est)

    print("Best fit parameters:")
    print("beta: ", beta_est)
    print("gamma:", gamma_est)
    print("sigma:", sigma_est)

    final_thetas = [beta_est, gamma_est, sigma_est]
    X0_final = np.array([logE0_est, logI0_est, logR0_est])
    return final_thetas, X0_final, res.fun
