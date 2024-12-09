from tfpigp.evaluation import *

# True parameter values used in simulations
true_params = [6.0, 0.6, 1.8]  # [beta, gamma, sigma]

# Define observed time points based on simulation settings
d_obs = 20          # Observations per unit time
t_max = 2.0         # Observation interval length
num_observations = int(d_obs * t_max) + 1
observed_time_points = np.linspace(0, t_max, num_observations)

# Path to the directory containing all simulation result subdirectories
results_dir = "../small run/"  # Replace with your actual path

# Generate the summary DataFrame
summary_df = summarize_simulation_results(results_dir, true_params, observed_time_points)

# Display the summary table
print("Summary of Simulation Results:")
print(summary_df)

# Save the summary to a CSV file for future reference
summary_df.to_csv("summary_simulation_results.csv", index=False)

# Compute and display mean RMSE across all simulations
mean_rmse_logE = summary_df["RMSE_logE"].mean()
mean_rmse_logI = summary_df["RMSE_logI"].mean()
mean_rmse_logR = summary_df["RMSE_logR"].mean()

mean_beta_error = summary_df["Beta_Error"].mean()
mean_gamma_error = summary_df["Gamma_Error"].mean()
mean_sigma_error = summary_df["Sigma_Error"].mean()

mean_beta_rmse = (summary_df["Beta_Error"]**2).mean()
mean_gamma_rmse = (summary_df["Gamma_Error"]**2).mean()
mean_sigma_rmse = (summary_df["Sigma_Error"]**2).mean()

print("\nMean Parameter RMSE Across Simulations:")
print(f"  Beta_RMSE: {mean_beta_rmse:.4f}")
print(f"  Gamma_RMSE: {mean_gamma_rmse:.4f}")
print(f"  Sigma_RMSE: {mean_sigma_rmse:.4f}")
