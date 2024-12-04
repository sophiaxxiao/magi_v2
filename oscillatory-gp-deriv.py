import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from IPython.display import clear_output

# core MAGI-TFP class
import tfpigp.magi_v2 as magi_v2

# 3-component model governing the SEIR system, appropriate for tensorflow vectorization
def f_vec(t, X, thetas):
    '''
    1. X - array containing (E, I, R) components. Suppose it is (N x D) for vectorization.
    2. theta - array containing (beta, gamma, sigma) components.
    3. Note that N_pop = 1.0, and that S is deterministic: S = 1 - (E + I + R)
    '''
    # implicitly compute S
    S = 1.0 - tf.reshape(tf.reduce_sum(X, axis=1), shape=(-1, 1))
    return tf.concat([(thetas[0] * S * X[:,1:2]) - (thetas[2] * X[:,0:1]), # dE/dt = bSI - sE
                      (thetas[2] * X[:,0:1]) - (thetas[1] * X[:,1:2]), # dI/dt = sE - gI
                      (thetas[1] * X[:,1:2])], # dR/dt = g*I
                     axis=1)

# initial data settings
d_obs = 20 # no. of observations per unit time
t_max = 2.0 # length of observation interval

# load in our data, thinning based on density of observations
raw_data = pd.read_csv('tfpigp/data/SEIR_beta=6_gamma=0.6_sigma=1.8_alpha=0.05_seed=2.csv').query(f"t <= {t_max}")
obs_data = raw_data.iloc[::int((raw_data.index.shape[0] - 1) / (d_obs * t_max))]

# extract out the time vector + noisy observations
ts_obs = obs_data.t.values.astype(np.float64)

# get the noisy observations
X_obs = obs_data[["E_obs", "I_obs", "R_obs"]].to_numpy().astype(np.float64) # S is implicit!

# create our model - f_vec is the ODE governing equations function defined earlier.
model = magi_v2.MAGI_v2(D_thetas=3, ts_obs=ts_obs, X_obs=X_obs, bandsize=None, f_vec=f_vec)

# fit Matern kernel hyperparameters (phi1, phi2) as well as (Xhat_init, sigma_sqs_init, thetas_init)
model.initial_fit(discretization=2, verbose=True)

# clear console for pretty output
clear_output(wait=True)

# collect our samples from NUTS posterior sampling - toggle tempering=True to use log-tempering.
results = model.predict(num_results=10, num_burnin_steps=10, tempering=False, verbose=True)

# visualize our trajectories
fig, ax = plt.subplots(1, 3, dpi=200, figsize=(10, 3))

# get our timesteps, mean trajectory predictions, and 2.5% + 97.5% trajectory predictions
I = results["I"].flatten()
Xhat_means = results["X_samps"].mean(axis=0)
Xhat_intervals = np.quantile(results["X_samps"], q=[0.025, 0.975], axis=0)
Xhat_init = results["Xhat_init"]

# go through each component and plot
for i, comp in enumerate(["$E$", "$I$", "$R$"]):

    # plot the ground truth + noisy observations
    ax[i].plot(raw_data["t"], raw_data[["E_true", "I_true", "R_true"][i]], color="black")

    # plot mean trajectory + 95% predictive interval
    ax[i].plot(I, Xhat_means[:, i], color="blue")
    ax[i].fill_between(I, Xhat_intervals[0, :, i], Xhat_intervals[1, :, i], alpha=0.3)
    ax[i].plot(I, Xhat_init[:, i], linestyle="--", color="green")
    ax[i].set_title(comp);
    ax[i].set_xlabel("$t$")
    ax[i].grid()

# shared legend + beautify
observed_components_desc = str(tuple(np.array(["$E$", "$I$", "$R$"]))).replace("'", "").strip()
plt.suptitle(f"Predicted Trajectories vs. Ground Truth | Observed Components: {observed_components_desc}")
custom_lines = [Line2D([0], [0], color="black", linewidth=1.0, alpha=1.0, label="Ground Truth"),
                Line2D([0], [0], color="grey", marker="o", linestyle="None", label="Noisy Observations"),
                Line2D([0], [0], color="blue", linewidth=1.0, alpha=1.0, label="Mean Prediction"),
                Line2D([0], [0], color="green", linestyle="--", linewidth=1.0, alpha=1.0, label="Initialization"),
                Patch(facecolor="blue", alpha=0.3, label="95% Predictive Interval")]
fig.legend(handles=custom_lines, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.075))
plt.tight_layout()
plt.show()

# check our parameters too
mean_thetas_pred = results["thetas_samps"].mean(axis=0)

print("Estimated Parameters:")
print(f"- Beta: {np.round(mean_thetas_pred[0], 3)} (Predicted) vs. 6.0 (Actual).")
print(f"- Gamma: {np.round(mean_thetas_pred[1], 3)} (Predicted) vs. 0.6 (Actual).")
print(f"- Sigma: {np.round(mean_thetas_pred[2], 3)} (Predicted) vs. 1.8 (Actual).")

# checking derivatives (i.e., physics fidelity)
raw_data["t"] = np.round(raw_data["t"].values, 3)
raw_data.set_index("t", inplace=True)

# get our true values
# TODO bug here: rounding error
X_plot = raw_data.loc[np.round(I, 3)][["E_true", "I_true", "R_true"]].values
X_plot = pd.read_csv('~/Downloads/xinfer.csv', index_col=0).values
thetas_true = np.array([6.0, 0.6, 1.8])

# compute GP-implied derivatives at truth
X_cent = tf.reshape(X_plot - model.mu_ds, shape=(X_plot.shape[0], 1, X_plot.shape[1]))
f_gp = model.m_ds @ tf.transpose(X_cent, perm=[2, 0, 1])

# compute the true derivatives at truth
f_ode = tf.transpose(f_vec(I, X_plot, thetas_true)[:, None], perm=[2, 0, 1])

# Compute finite difference derivatives
f_fd = []
delta_t = I[1] - I[0]  # Assuming uniform time intervals
for i in range(X_plot.shape[1]):  # Loop over components
    finite_diff = np.gradient(X_plot[:, i], delta_t)
    f_fd.append(finite_diff)
f_fd = np.array(f_fd)

# Plot the results
fig, ax = plt.subplots(1, 4, dpi=200, figsize=(12, 3))

# Plot ODE vs. GP-implied vs. Finite Difference derivatives for each component
for i, comp in enumerate(["$E$", "$I$", "$R$"]):
    ax[i].set_title(comp)
    ax[i].plot(I, f_gp[i], label="GP", linestyle="-")
    ax[i].plot(I, f_ode[i], label="ODE", linestyle="--")
    ax[i].plot(I, f_fd[i], label="Finite Difference", linestyle=":")
    ax[i].grid()
    ax[i].legend()

# Also check the log-posterior
ax[3].set_title("Log-Posterior")
ax[3].plot(results["kernel_results"].inner_results.target_log_prob)
ax[3].grid()

# Beautify
plt.tight_layout()
plt.show()

# Original E component
E_true = X_plot[:, 0]

# Finite Difference Derivative for E
E_fd = np.gradient(E_true, delta_t)

# GP-implied derivative for E
E_gp = f_gp[0].numpy()

# ODE derivative for E
E_ode = f_ode[0].numpy()

# Plotting
fig, ax = plt.subplots(2, 1, dpi=200, figsize=(10, 6), sharex=True)

# Original E values
ax[0].plot(
    I, E_true,
    color='blue', label='Original E (level)',
    linestyle='-', marker='o', markersize=2, markerfacecolor='red', markeredgecolor='black'
)
ax[0].set_title('Original E Component (Level)')
ax[0].set_ylabel('E')
ax[0].grid()
ax[0].legend()

# Derivatives of E
ax[1].plot(I, E_gp, color='blue', label='GP Derivative')
ax[1].plot(I, E_ode, color='red', label='ODE Derivative')
ax[1].plot(I, E_fd, color='green', label='Finite Difference Derivative')
ax[1].set_title('Derivatives of E Component')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('dE/dt')
ax[1].grid()
ax[1].legend()

# Adjust layout and show
plt.tight_layout()
plt.show()
