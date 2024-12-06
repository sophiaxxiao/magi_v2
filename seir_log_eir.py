import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tfpigp.magi_v2 as magi_v2  # MAGI-TFP class for Bayesian inference
from tfpigp.visualization import *

# Define the EIR representation ODE on the log scale
def f_vec(t, X, thetas):
    '''
    Log-scale SEIR model with E implicitly represented.

    Parameters:
    1. X - array containing (logE, logI, logR) components. Shape (N x 3).
    2. thetas - array containing (beta, gamma, sigma) parameters.

    Returns:
    Derivatives (dlogE/dt, dlogI/dt, dlogR/dt) as a tensor of shape (N x 3).
    '''
    logE, logI, logR = tf.unstack(X, axis=1)
    beta = thetas[0]
    gamma = thetas[1]
    sigma = thetas[2]

    # Convert log variables back to original scale
    E = tf.exp(logE)
    I = tf.exp(logI)
    R = tf.exp(logR)

    # Compute S implicitly (EIR representation)
    S = 1.0 - E - I - R

    # Ensure stability for S
    S = tf.maximum(S, 1e-10)

    # Derivatives in the original scale
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

    # Derivatives in the log scale (chain rule)
    dlogEdt = dEdt / E
    dlogIdt = dIdt / I
    dlogRdt = dRdt / R

    # Return derivatives as a tensor
    return tf.stack([dlogEdt, dlogIdt, dlogRdt], axis=1)

# Initial settings
d_obs = 20  # Observations per unit time
t_max = 2.0  # Observation interval length

# Load data and select observations
raw_data = pd.read_csv('tfpigp/data/logSEIR_beta=6.0_gamma=0.6_sigma=1.8_alpha=0.15_seed=2.csv').query(f"t <= {t_max}")
obs_data = raw_data.iloc[::int((raw_data.index.shape[0] - 1) / (d_obs * t_max))]

# Extract observation times and log-transformed noisy observations
ts_obs = obs_data.t.values.astype(np.float64)
X_obs = np.log(obs_data[["E_obs", "I_obs", "R_obs"]].to_numpy().astype(np.float64))

# Create the MAGI-TFP model
model = magi_v2.MAGI_v2(D_thetas=3, ts_obs=ts_obs, X_obs=X_obs, bandsize=None, f_vec=f_vec)

# Fit initial hyperparameters
phi_exo = None
model.initial_fit(discretization=2, verbose=True, use_fourier_prior=False, phi_exo=phi_exo)
model.phi1s
model.phi2s
model.sigma_sqs_init

clear_output(wait=True)

# Collect samples using NUTS posterior sampling
results = model.predict(num_results=1000, num_burnin_steps=1000, tempering=False, verbose=True)

ts_true = raw_data.t.values
x_true = raw_data[["E_true", "I_true", "R_true"]]
x_true = np.log(x_true)
plot_trajectories(ts_true, x_true, results, ts_obs, X_obs)
plot_trajectories(ts_true, x_true, results, ts_obs, X_obs, trans_func=np.exp)
print_parameter_estimates(results, [6.0, 0.6, 1.8])
plot_trace(results["thetas_samps"], [6.0, 0.6, 1.8], ["beta", "gamma", "sigma"])
