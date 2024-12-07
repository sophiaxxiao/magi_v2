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
# Load and prepare data
##############################

# Adjust as needed
raw_data = pd.read_csv('tfpigp/data/logSEIR_beta=6.0_gamma=0.6_sigma=1.8_alpha=0.15_seed=2.csv')
t_max = 2.0
d_obs = 20
raw_data = raw_data.query(f"t <= {t_max}")

# Pick observation times
obs_data = raw_data.iloc[::int((raw_data.index.shape[0] - 1) / (d_obs * t_max))]

ts_obs = obs_data['t'].values.astype(np.float64)

# Observations: log(E_obs), log(I_obs), log(R_obs)
X_obs_full = np.log(obs_data[["E_obs", "I_obs", "R_obs"]].to_numpy().astype(np.float64))
# E is not actually observed, we simulate that by setting E_obs to NaN:
X_obs_full[:, 0] = np.nan

# We'll fit to I and R only
logI_obs = X_obs_full[:,1]
logR_obs = X_obs_full[:,2]

##############################
# Negative log-likelihood function
##############################
def negative_log_likelihood(params):
    # params = [log_beta, log_gamma, log_sigma, logE0, logI0, logR0]
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

    X_pred = sol.y.T  # shape: (len(ts_obs), 3)

    logI_pred = X_pred[:,1]
    logR_pred = X_pred[:,2]

    mask_I = ~np.isnan(logI_obs)
    mask_R = ~np.isnan(logR_obs)

    residuals_I = (logI_obs[mask_I] - logI_pred[mask_I])**2
    residuals_R = (logR_obs[mask_R] - logR_pred[mask_R])**2

    # Assume known measurement noise variance on log scale
    sigma_obs = 0.1
    var_obs = sigma_obs**2

    sse = np.sum(residuals_I) + np.sum(residuals_R)
    # Negative log-likelihood (up to constant): sse/(2*var_obs)
    nll = sse/(2*var_obs)

    return nll

##############################
# Run optimization
##############################

# True parameters: beta=6.0, gamma=0.6, sigma=1.8
# We do not know initial states either. Let's pick initial guesses:
# For parameters, start near true values:
init_log_beta = 0.5*np.random.randn()
init_log_gamma = 0.5*np.random.randn()
init_log_sigma = 0.5*np.random.randn()

# Initial states: we can guess from data:
# If we trust the first observation for I and R roughly:
init_logI0 = logI_obs[0] if not np.isnan(logI_obs[0]) else -4.0
init_logR0 = logR_obs[0] if not np.isnan(logR_obs[0]) else -4.0
# E0 not observed, guess small:
init_logE0 = -4.0

init_guess = [init_log_beta, init_log_gamma, init_log_sigma, init_logE0, init_logI0, init_logR0]

bounds = [(-3, 3), (-3, 3), (-3, 3), (-10, 0), (-10, 5), (-10, 5)]
# Bounds are arbitrary. Adjust if needed.

res = minimize(negative_log_likelihood, init_guess, method='L-BFGS-B',
               bounds=bounds, options={'maxiter':1000})

est_params = res.x
log_beta_est, log_gamma_est, log_sigma_est, logE0_est, logI0_est, logR0_est = est_params
beta_est = np.exp(log_beta_est)
gamma_est = np.exp(log_gamma_est)
sigma_est = np.exp(log_sigma_est)

print("Best fit parameters:")
print("beta: ", beta_est)
print("gamma:", gamma_est)
print("sigma:", sigma_est)
print("logE0:", logE0_est)
print("logI0:", logI0_est)
print("logR0:", logR0_est)
print("NLL:  ", res.fun)

##############################
# Check identifiability by multiple starts
##############################

num_starts = 5
solutions = []
for i in range(num_starts):
    # random start
    init_guess_random = [
        np.log(6.0) + 0.5*np.random.randn(),
        np.log(0.6) + 0.5*np.random.randn(),
        np.log(1.8) + 0.5*np.random.randn(),
        -4.0 + np.random.randn(),
        init_logI0 + 0.5*np.random.randn(),
        init_logR0 + 0.5*np.random.randn()
    ]
    res_rand = minimize(negative_log_likelihood, init_guess_random, method='L-BFGS-B',
                        bounds=bounds, options={'maxiter':1000})
    solutions.append(res_rand.x)

solutions = np.array(solutions)
print("\nMultiple solutions from random starts (log-scale):")
print(solutions)
print("Mean solution (log-scale):", solutions.mean(axis=0))
print("Std of solutions (log-scale):", solutions.std(axis=0))

##############################
# Visualization
##############################

# Generate predictions from the optimized parameters
final_thetas = [beta_est, gamma_est, sigma_est]
X0_final = np.array([logE0_est, logI0_est, logR0_est])
sol_final = solve_ivp(lambda t, y: ODE_log_scale(t, y, final_thetas),
                      t_span=(ts_obs[0], ts_obs[-1]),
                      y0=X0_final, t_eval=ts_obs, rtol=1e-10, atol=1e-10)

X_pred_final = sol_final.y.T
logI_pred_final = X_pred_final[:,1]
logR_pred_final = X_pred_final[:,2]

# Plot observed vs predicted (I and R)
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs[0].plot(ts_obs, logI_obs, 'o', label='Observed log(I)')
axs[0].plot(ts_obs, logI_pred_final, '-', label='Predicted log(I)')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('log(I)')
axs[0].legend()

axs[1].plot(ts_obs, logR_obs, 'o', label='Observed log(R)')
axs[1].plot(ts_obs, logR_pred_final, '-', label='Predicted log(R)')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('log(R)')
axs[1].legend()

plt.tight_layout()
plt.show()

# Additionally, we can check identifiability visually by plotting the distribution
# of parameter estimates from different random starts
fig, axs = plt.subplots(3, 2, figsize=(10, 10))
param_names = ["log_beta", "log_gamma", "log_sigma", "logE0", "logI0", "logR0"]
for i, ax in enumerate(axs.flatten()):
    ax.hist(solutions[:, i], bins=10, alpha=0.7, color='skyblue')
    ax.axvline(solutions[:, i].mean(), color='red', linestyle='--', label='mean')
    ax.set_title(param_names[i])
    ax.legend()

plt.tight_layout()
plt.show()
