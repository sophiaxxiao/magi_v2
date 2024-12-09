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

np.abs(summary_df.iloc[:, 1:]).mean()
np.abs(summary_df.iloc[:, 1:]).std()

summary_df_print = summary_df.copy()
summary_df_print.columns = [x.replace('_', ' ') for x in summary_df_print.columns]

generate_latex_table(summary_df_print, "summary_table.tex")
