from Stochastic_Core_Library.quad_var_cov_estimator import geometric_bm
import numpy as np
import pandas as pd
import scipy.stats as stats

def discretisation_bridge_monitoring(s_0: float, strike: float, sd: float, T: float, rate: float, n: int, n_sims: int, barrier_raw: float):
    prob = np.zeros((n-1, n_sims))
    dt = T / (n-1)

    barrier = np.log(barrier_raw)
    gbm_prices = geometric_bm(s_0, sd, T, rate, n, n_sims)
    naive_hits = (gbm_prices >= barrier_raw).any(axis=0).astype(int)

    simulation = np.log(gbm_prices.to_numpy())
    for i in range(n_sims):
        for j in range(n-1):
            if simulation[j,i] < barrier and simulation[j+1,i] < barrier:
                prob[j, i] = np.exp(-(2*(barrier - simulation[j,i]) * (barrier - simulation[j+1,i])) / (sd**2 * dt))
            else:
                prob[j, i] = 1

    rand_uniform = stats.uniform.rvs(0, 1, np.shape(prob))
    path_simulations = (prob > rand_uniform).any(axis=0).astype(int)
    path_survival_prob = (1 - prob).prod(axis=0)

    weighted_payoff = path_survival_prob * np.maximum(gbm_prices.iloc[-1, :] - strike, 0)

    pd.set_option('display.float_format', '{:.4f}'.format)
    prob = pd.DataFrame(prob, columns=np.arange(1, n_sims+1, 1))
    survival_prob = pd.DataFrame(path_survival_prob)
    return gbm_prices, naive_hits, prob, survival_prob.T, path_simulations, weighted_payoff

prices, naive, probs, surv_probs, path_sim, weighted_payoff = discretisation_bridge_monitoring(1000, 955, 0.2**0.5, 1, 0.1, 1000, 1000, 1200)
print("---------------------------- Prices ----------------------------")
print(prices)
print("---------------------------- Path Hit Probabilities ----------------------------")
print(probs)
print("---------------------------- Naive Discrete Monitoring ----------------------------")
print(naive)
print("---------------------------- Survival Hit Probabilities ----------------------------")
print(surv_probs)
print("---------------------------- Path Simulator ----------------------------")
print(path_sim)
print("---------------------------- Path Hit Probability Counts----------------------------")
print(f"{probs[(probs != 0) & (probs != 1)].count().sum()} possible hits between the sampled points")
print(f"{probs[probs == 0].count().sum()} times there is no chance of hitting the barrier on intervals")
print(f"{probs[probs == 1].count().sum()} times certain there is a hit on intervals")
print("---------------------------- Path Survival Probability Counts ----------------------------")
print(f"{surv_probs[(surv_probs != 0) & (surv_probs != 1)].count().sum()} possible hits between the sampled points")
print(f"{surv_probs[surv_probs == 0].count().sum()} times there is no chance of hitting the barrier on intervals")
print(f"{surv_probs[surv_probs == 1].count().sum()} times certain there is a hit on intervals")
print(weighted_payoff)
