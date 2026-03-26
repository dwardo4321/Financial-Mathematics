from Stochastic_Core_Library.quad_var_cov_estimator import geometric_bm
import numpy as np
import pandas as pd

def discretisation_monitoring(s_0: float, sd: float, t_n: float, rate: float, n_all: int, n_sims: int, barrier_raw: float):
    prob = np.zeros((n_all-1, n_sims))
    dt = t_n / (n_all-1)

    barrier = np.log(barrier_raw)
    gbm_prices = geometric_bm(s_0, sd, t_n, rate, n_all, n_sims)
    simulation = np.log(gbm_prices.to_numpy())
    for i in range(n_sims):
        for j in range(n_all-1):
            if simulation[j,i] < barrier and simulation[j+1,i] < barrier:
                prob[j, i] = np.exp(-(2*(barrier - simulation[j,i]) * (barrier - simulation[j+1,i])) / (sd**2 * dt))
            else:
                prob[j, i] = 1

    pd.set_option('display.float_format', '{:.4f}'.format)
    prob = pd.DataFrame(prob, columns=np.arange(1, n_sims+1, 1))
    return gbm_prices, prob

prices, probs = discretisation_monitoring(1000, 0.2**0.5, 1, 0.1, 1000, 1000, 1000)
print(prices)
print(probs)
print(f"{probs[(probs != 0) & (probs != 1)].count().sum()} possible hits between the sampled points")
print(f"{probs[probs == 0].count().sum()} times there is no chance of hitting the barrier on intervals")
print(f"{probs[probs == 1].count().sum()} times certain there is a hit on intervals")
