import os
import pickle
import numpy as np
import pandas as pd


def compute_rmse(true_values, inferred_samples, observed_indices):
    """
    Compute the Root Mean Square Error (RMSE) between the true values and inferred samples.

    Parameters:
    - true_values (np.array): Shape (T, D), true trajectories.
    - inferred_samples (np.array): Shape (num_samples, T, D), posterior samples.
    - observed_indices (np.array): Indices of observed time points.

    Returns:
    - float: RMSE value.
    """
    # Extract true values at observed points
    true_obs = true_values[observed_indices]

    # Compute the mean inferred trajectory
    inferred_mean = inferred_samples.mean(axis=0)[observed_indices]

    # Calculate RMSE
    mse = np.mean((true_obs - inferred_mean) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def compute_log_rmse(true_values_log, inferred_samples):
    """
    Compute RMSE on the log scale.

    Parameters:
    - true_values_log (np.array): Shape (T, D), true log-transformed trajectories.
    - inferred_samples (np.array): Shape (num_samples, T, D), posterior samples.
    - observed_indices (np.array): Indices of observed time points.

    Returns:
    - float: RMSE on log scale.
    """
    # Extract true log values at observed points
    true_obs_log = true_values_log

    # Compute the mean inferred log trajectory
    inferred_mean_log = inferred_samples.mean(axis=0)

    # Calculate RMSE
    mse_log = np.mean((true_obs_log - inferred_mean_log) ** 2, axis=0)
    rmse_log = np.sqrt(mse_log)
    return rmse_log


def compute_parameter_error(true_params, inferred_samples):
    """
    Compute parameter estimation errors.

    Parameters:
    - true_params (list or np.array): True parameter values.
    - inferred_samples (np.array): Shape (num_samples, P), posterior samples of parameters.

    Returns:
    - np.array: Absolute errors for each parameter.
    """
    inferred_mean = inferred_samples.mean(axis=0)
    errors = inferred_mean - np.array(true_params)
    return errors


def compute_parameter_rmse(true_params, inferred_samples):
    """
    Compute RMSE for parameter estimation.

    Parameters:
    - true_params (list or np.array): True parameter values.
    - inferred_samples (np.array): Shape (num_samples, P), posterior samples of parameters.

    Returns:
    - np.array: RMSE for each parameter.
    """
    inferred_mean = inferred_samples.mean(axis=0)
    mse = np.mean((inferred_mean - np.array(true_params)) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def summarize_simulation_results(results_dir, true_params, observed_time_points):
    """
    Summarize simulation results across multiple pickle files.

    Parameters:
    - results_dir (str): Path to the directory containing simulation result subdirectories.
    - true_params (list or np.array): True parameter values [beta, gamma, sigma].
    - observed_time_points (np.array): Array of observed time points.

    Returns:
    - pd.DataFrame: Summary table with RMSE and parameter errors for each simulation.
    """
    summary = []

    # Iterate through each simulation directory
    for sim_dir in os.listdir(results_dir):
        sim_path = os.path.join(results_dir, sim_dir)
        if os.path.isdir(sim_path):
            pickle_file = os.path.join(sim_path, "simulation_results.pkl")
            if os.path.exists(pickle_file):
                with open(pickle_file, "rb") as f:
                    data = pickle.load(f)

                # Extract necessary data
                ts_true = data["ts_true"]  # Shape (T,)
                x_true = data["x_true"]  # Shape (T, D)
                results_forecast = data["results_forecast"]
                X_obs = data["X_obs"]  # Shape (observed_T, D)
                results = data["results"]  # Contains "thetas_samps", "X_samps", etc.

                # Identify observed time indices
                observed_indices_in_true = np.isin(np.round(ts_true, 5), np.round(observed_time_points, 5)).nonzero()[0]
                observed_indices_in_I = np.isin(np.round(results['I'], 5), np.round(observed_time_points, 5)).nonzero()[0]

                # Compute RMSE on log scale
                X_samps = results["X_samps"]  # Shape (num_samples, T, D)
                rmse_log = compute_log_rmse(x_true.loc[observed_indices_in_true, :], X_samps[:, observed_indices_in_I, :])

                # Compute parameter estimation errors
                thetas_samps = results["thetas_samps"]  # Shape (num_samples, P)
                param_errors = compute_parameter_error(true_params, thetas_samps)

                # Append results to summary
                summary.append({
                    "Simulation": sim_dir,
                    "RMSE_log": rmse_log,
                    "Beta_Error": param_errors[0],
                    "Gamma_Error": param_errors[1],
                    "Sigma_Error": param_errors[2],
                })

    # Convert summary to DataFrame
    summary_df = pd.DataFrame(summary)
    return summary_df
