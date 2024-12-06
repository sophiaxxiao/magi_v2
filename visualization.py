import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(ts_true, x_true, results, ts_obs, X_obs, trans_func=lambda x: x):
    """
    Plot the predicted trajectories against ground truth and noisy observations.

    Parameters:
    - ts_true: Array of time steps for the ground truth data.
    - x_true: DataFrame or array of ground truth data (columns for each component, e.g., ["E", "I", "R"]).
    - results: Dictionary containing prediction results from MAGI-TFP.
    - ts_obs: Array of time steps corresponding to noisy observations.
    - X_obs: Array of noisy observations.
    - trans_func: Transformation function to apply to all plotted data (default: identity function).
    """
    fig, ax = plt.subplots(1, x_true.shape[1], dpi=200, figsize=(12, 6))
    I = results["I"].flatten()
    Xhat_means = trans_func(results["X_samps"].mean(axis=0))
    Xhat_intervals = np.quantile(trans_func(results["X_samps"]), q=[0.025, 0.975], axis=0)
    Xhat_init = trans_func(results["Xhat_init"])

    for i in range(x_true.shape[1]):
        comp = x_true.columns[i]
        true_values = trans_func(x_true[comp].values)

        # Plot ground truth trajectory
        ax[i].plot(ts_true, true_values, color="black", label="Ground Truth")
        # Plot mean trajectory and predictive interval
        ax[i].plot(I, Xhat_means[:, i], color="blue", label="Mean Prediction")
        ax[i].fill_between(I, Xhat_intervals[0, :, i], Xhat_intervals[1, :, i], color="blue", alpha=0.3, label="95% Predictive Interval")
        ax[i].plot(I, Xhat_init[:, i], linestyle="--", color="green", label="Initialization")
        # Plot noisy observations
        ax[i].scatter(ts_obs, trans_func(X_obs[:, i]), color="grey", s=20, zorder=5, label="Noisy Observations")
        # Titles and labels
        ax[i].set_title(f"Component {comp}")
        ax[i].set_xlabel("$t$")
        ax[i].set_ylabel(f"{comp}")
        ax[i].grid()

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_trace(thetas_samps, param_names=["$\\beta$", "$\\gamma$", "$\\sigma$"]):
    """
    Plot trace plots for sampled parameters.

    Parameters:
    - thetas_samps: Array of sampled parameters with shape (num_samples, num_params).
    - param_names: List of parameter names for labeling (default: ["$\\beta$", "$\\gamma$", "$\\sigma$"]).
    """
    fig, ax = plt.subplots(len(param_names), 1, figsize=(10, 8), dpi=200, sharex=True)

    for i, param in enumerate(param_names):
        ax[i].plot(thetas_samps[:, i], alpha=0.7, lw=1)
        ax[i].set_title(f"Trace Plot for {param}", fontsize=12)
        ax[i].set_ylabel("Value")
        ax[i].grid()

    ax[-1].set_xlabel("Iteration")
    plt.tight_layout()
    plt.show()


def print_parameter_estimates(results, true_values):
    """
    Print the estimated parameters against the true values.

    Parameters:
    - results: Dictionary containing prediction results from MAGI-TFP.
    - true_values: List of true parameter values [beta, gamma, sigma].
    """
    mean_thetas_pred = results["thetas_samps"].mean(axis=0)
    print("Estimated Parameters:")
    for i, (predicted, actual) in enumerate(zip(mean_thetas_pred, true_values)):
        print(f"- Parameter {i + 1}: {np.round(predicted, 3)} (Predicted) vs. {actual} (Actual).")

