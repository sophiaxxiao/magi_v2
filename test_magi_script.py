# Script to test MAGI Inference w/ SIRW Model 

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from IPython.display import clear_output
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# core MAGI-TFP class
import magi_v2


# 4-component model governing the SIRW system, appropriate for tensorflow vectorization
def f_vec(t, X, thetas):
    """
    Computes the time derivative of the SIRW model components.

    Args:
        t: Time (not used in this function, but present for consistency in ODE frameworks).
        X: Tensor of shape (N, 4), containing (S, I, R, W) components.
        thetas: Tensor of shape (5,), containing model parameters (beta, phi, xi, chi, kappa).
    
    Returns:
        Tensor of shape (N, 4), the time derivatives of (S, I, R, W).
    """
    # Extract individual components from the state vector
    S, I, R, W = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]
    beta, phi, xi, chi, kappa = thetas[0], thetas[1], thetas[2], thetas[3], thetas[4]

    # Compute the derivatives
    dS_dt = -beta * S * I + kappa * W           # dS/dt = -beta * S * I + kappa * W
    dI_dt = beta * S * I - phi * I              # dI/dt = beta * S * I - phi * I
    dR_dt = phi * I - xi * R + chi * I * W      # dR/dt = phi * I - xi * R + chi * I * W
    dW_dt = xi * R - chi * I * W - kappa * W    # dW/dt = xi * R - chi * I * W - kappa * W

    # Concatenate derivatives into a single tensor
    return tf.concat([dS_dt, 
                      dI_dt, 
                      dR_dt, 
                      dW_dt], axis=1)

# load in our data, thinning based on density of observations
raw_data = pd.read_csv(f"/local/scratch/sxxiao/GP-SIR-Experiments/Experiment_3_SIRW/data/sirw_endemic_beta0.3_phi0.1_xi0.01_chi0.1_kappa0.01_clean.csv")


# initial data settings
d_obs = 1 # no. of observations per unit time
t_max = 3 * 365 # we have observations from T=0 to T= 3 * 365.
comp_obs = [True, True, True, True] # which components are observed? Here, everything is observed.

 
obs_data = raw_data.iloc[::int((raw_data.index.shape[0] - 1) / (d_obs * t_max))] # Select every step-th row from raw_data, where step is the integer result of the division.

# extract out the time vector + noisy observations
ts_obs = obs_data.t.values.astype(np.float64) 

X_obs = obs_data[["S", "I", "R", "W"]].to_numpy().astype(np.float64)

X_obs[X_obs < 0.0] = 0.0 # we know SIRW must be between [0, 1]

# make certain components missing if necessary
for i, comp_obs_val in enumerate(comp_obs):
    if comp_obs_val != True:
        X_obs[:,i] = np.nan

# create our model - f_vec is the ODE governing equations function defined earlier.
model = magi_v2.MAGI_v2(D_thetas=3, ts_obs=ts_obs, X_obs=X_obs, bandsize=200, f_vec=f_vec)

# fit Matern kernel hyperparameters (phi1, phi2) as well as (Xhat_init, sigma_sqs_init, thetas_init)
model.initial_fit(discretization=1, verbose=True)

# clear console for pretty output
clear_output(wait=True)

# collect our samples from NUTS posterior sampling
results = model.predict(num_results=1000, num_burnin_steps=1000, verbose=True)

# Save the DataFrame to a CSV file
results_df.to_csv('/results.csv', index=False)

# Save results 
print("Results saved as CSV!")