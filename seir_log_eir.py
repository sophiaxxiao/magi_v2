import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tfpigp.magi_v2 as magi_v2  # MAGI-TFP class for Bayesian inference

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
raw_data = pd.read_csv('tfpigp/data/logSEIR_beta=6.0_gamma=0.6_sigma=1.8_alpha=0.05_seed=2.csv').query(f"t <= {t_max}")
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

# Visualize trajectories
fig, ax = plt.subplots(1, 3, dpi=200, figsize=(12, 6))
I = results["I"].flatten()
Xhat_means = results["X_samps"].mean(axis=0)
Xhat_intervals = np.quantile(results["X_samps"], q=[0.025, 0.975], axis=0)
Xhat_init = results["Xhat_init"]

for i, comp in enumerate(["$E$", "$I$", "$R$"]):
    # Plot ground truth trajectory
    ax[i].plot(raw_data["t"], np.log(raw_data[["E_true", "I_true", "R_true"][i]]), color="black", label="Ground Truth")
    # Plot mean trajectory and predictive interval
    ax[i].plot(I, Xhat_means[:, i], color="blue", label="Mean Prediction")
    ax[i].fill_between(I, Xhat_intervals[0, :, i], Xhat_intervals[1, :, i], color="blue", alpha=0.3, label="95% Predictive Interval")
    ax[i].plot(I, Xhat_init[:, i], linestyle="--", color="green", label="Initialization")
    # Plot noisy observations
    ax[i].scatter(ts_obs, X_obs[:, i], color="grey", s=20, zorder=5, label="Noisy Observations")
    ax[i].set_title(f"Component {comp}")
    ax[i].set_xlabel("$t$")
    ax[i].set_ylabel(f"{comp}")
    ax[i].grid()

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Print parameter estimates
mean_thetas_pred = results["thetas_samps"].mean(axis=0)
print("Estimated Parameters:")
print(f"- Beta: {np.round(mean_thetas_pred[0], 3)} (Predicted) vs. 6.0 (Actual).")
print(f"- Gamma: {np.round(mean_thetas_pred[1], 3)} (Predicted) vs. 0.6 (Actual).")
print(f"- Sigma: {np.round(mean_thetas_pred[2], 3)} (Predicted) vs. 1.8 (Actual).")

import matplotlib.pyplot as plt

# Extract the parameter samples
thetas_samps = results["thetas_samps"]
beta_samples = thetas_samps[:, 0]
gamma_samples = thetas_samps[:, 1]
sigma_samples = thetas_samps[:, 2]

# Create trace plots for each parameter
fig, ax = plt.subplots(3, 1, figsize=(10, 8), dpi=200, sharex=True)

# Beta trace plot
ax[0].plot(beta_samples, color='blue', alpha=0.7, lw=1)
ax[0].set_title("Trace Plot for $\\beta$ (Beta)", fontsize=12)
ax[0].set_ylabel("Value")
ax[0].grid()

# Gamma trace plot
ax[1].plot(gamma_samples, color='green', alpha=0.7, lw=1)
ax[1].set_title("Trace Plot for $\\gamma$ (Gamma)", fontsize=12)
ax[1].set_ylabel("Value")
ax[1].grid()

# Sigma trace plot
ax[2].plot(sigma_samples, color='red', alpha=0.7, lw=1)
ax[2].set_title("Trace Plot for $\\sigma$ (Sigma)", fontsize=12)
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("Value")
ax[2].grid()

# Adjust layout and show
plt.tight_layout()
plt.show()
