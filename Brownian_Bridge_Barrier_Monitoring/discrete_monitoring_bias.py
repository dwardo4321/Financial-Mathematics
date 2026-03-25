from Stochastic_Core_Library.quad_var_cov_estimator import geometric_bm
import numpy as np
import pandas as pd

def discretisation_monitoring(barrier_raw: float, n_sims: int, s_0: float, sd: float, t_n: float, rate: float, n_all: int, log_paths: bool=True):
    sim = np.zeros((n_all, n_sims))
    prob = np.zeros((n_all, n_sims))
    dt = t_n / (n_all-1)
    barrier = np.log(barrier_raw)
    for i in range(n_sims):
        if log_paths:
            simulation = np.log(geometric_bm(s_0, sd, t_n, rate, n_all).to_numpy())
        else:
            simulation = geometric_bm(s_0, sd, t_n, rate, n_all).to_numpy()

        sim[:, i] = simulation.flatten()
        for j in range(n_all-1):
            prob[j, i] = np.exp(-(2*(barrier - sim[j,i]) * (barrier - sim[j+1,i])) / sd**2 * dt)


    sim = pd.DataFrame(sim, columns = np.arange(1, n_sims+1, 1))
    prob = pd.DataFrame(prob, columns=np.arange(1, n_sims+1, 1))
    return sim, prob

print(discretisation_monitoring(6, 10,1000, 0.2, 20, 0.1, 1000))
