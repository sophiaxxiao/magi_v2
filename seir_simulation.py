import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# generate SEIR datasets
for seed in range(10):
    for alpha in [0.05, 0.15]:
        # parameter set for beta, gamma, sigma
        for params in [(6.0, 0.6, 1.8)]:
            np.random.seed(seed)

            b, g, sigma, alpha_value, N = params[0], params[1], params[2], alpha, 10000


            # encode ODEs for solve_ivp data-generation processes
            def SEIR(t, y):
                # unpack y
                S, E, I, R = tuple(y)

                # SEIR model equations
                dSdt = -b * S * I / N
                dEdt = b * S * I / N - sigma * E
                dIdt = sigma * E - g * I
                dRdt = g * I

                # return only the derivatives
                return np.array([dSdt, dEdt, dIdt, dRdt])


            # initial conditions
            SEIR_init = np.array([9899, 50, 50, 1])
            t_start, t_end = 0.0, 20.0
            t_steps = np.linspace(start=t_start, stop=t_end, num=20001)

            # solve the system
            X = solve_ivp(fun=SEIR, t_span=(t_start, t_end), y0=SEIR_init,
                          t_eval=t_steps, atol=1e-10, rtol=1e-10).y.T

            # Compute noise levels (on log scale)
            X_log = np.log(X)

            # Add noise to the data (on log scale)
            X_noised_log = X_log.copy()
            X_noised_log += np.random.normal(loc=0.0, scale=alpha_value, size=X_log.shape)
            X_noised = np.exp(X_noised_log)

            # save time, X_noised, and true values
            data = np.hstack([t_steps.reshape(-1, 1), X_noised, X])
            cols = ["t", "S_obs", "E_obs", "I_obs", "R_obs",
                    "S_true", "E_true", "I_true", "R_true"]

            df = pd.DataFrame(data=data, columns=cols)
            filename = f"tfpigp/data/logSEIR_beta={b}_gamma={g}_sigma={sigma}_alpha={alpha_value}_seed={seed}.csv"
            df.to_csv(filename, index=False)
