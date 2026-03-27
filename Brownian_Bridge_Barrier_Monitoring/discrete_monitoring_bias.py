from Stochastic_Core_Library.quad_var_cov_estimator import geometric_bm
from Utility_Functions.utilities_delta_gamma_engine import delta_engine_plotter
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def discretisation_bridge_monitoring(s_0: float, strike: float, sd: float, T: float, rate: float, n_lengths: list, n_sims: int, barrier_raw: float, plot: bool = False, debug: bool = False):

    n_lengths.append(n_lengths[-1] + 10000)  # for benchmark price

    monte_carlo_prices = pd.DataFrame(columns = ["dt", "Bridge Correction Price", "Naive Monitoring Price"], index = n_lengths)

    for k, n in enumerate(monte_carlo_prices.index):
        prob = np.zeros((n-1, n_sims))
        dt = T / (n-1)
        monte_carlo_prices.iloc[k, 0] = dt

        barrier = np.log(barrier_raw)
        gbm_prices = geometric_bm(s_0, sd, T, rate, n, n_sims)

    # Bridge Correction Monitoring -------------------------------------------------------------------------------------
        # Simulated Log Prices
        simulation = np.log(gbm_prices.to_numpy())

        # Hit Probabilities
        for i in range(n_sims):
            for j in range(n-1):
                if simulation[j,i] < barrier and simulation[j+1,i] < barrier:
                    prob[j, i] = np.exp(-(2*(barrier - simulation[j,i]) * (barrier - simulation[j+1,i])) / (sd**2 * dt))
                else:
                    prob[j, i] = 1

        prob = pd.DataFrame(prob, columns=np.arange(1, n_sims + 1, 1))

        # Bernoulli Path Generator
        rand_uniform = stats.uniform.rvs(0, 1, np.shape(prob))
        path_simulations = (prob > rand_uniform).any(axis=0).astype(int)

        # Path Survival Probabilities
        path_survival_prob = (1 - prob).prod(axis=0)

        # Survival Weighted Payoff
        maturity_survival_weighted_payoff = path_survival_prob * np.maximum(gbm_prices.iloc[-1, :] - strike, 0)

        # Monte Carlo Payoff
        monte_carlo_bridge_corrected_payoff = np.exp(-rate * T) * (1 / n_sims) * maturity_survival_weighted_payoff.sum()

        # Output
        if debug:
            pd.set_option('display.float_format', '{:.4f}'.format)
            print()
            print(f"-----#####-----#####-----#####-----#####----- Bridge Correction Monitoring, n = {n} -----#####-----#####-----#####-----#####-----\n")
            print({"Log Price Simulations\n": simulation, "Hit Probabilities\n": prob, "Bernoulli Path Simulations\n": path_simulations,
                   "Path Survival Probabilities\n": path_survival_prob, "Survival Weighted Payoff\n": maturity_survival_weighted_payoff,
                   "Monte Carlo Bridge Corrected Payoff\n": monte_carlo_bridge_corrected_payoff})

        monte_carlo_prices.iloc[k, 1] = monte_carlo_bridge_corrected_payoff

    # Naive Discrete Monitoring ----------------------------------------------------------------------------------------
        # Naive Hits
        naive_hits = (gbm_prices >= barrier_raw).any(axis=0).astype(int)

        # Naive Knock-Out Payoff
        naive_ko_payoff = (1-naive_hits) * np.maximum(gbm_prices.iloc[-1, :] - strike, 0)

        # Naive Monte Carlo Payoff
        naive_monte_carlo_payoff = np.exp(-rate * T) * (1 / n_sims) * naive_ko_payoff.sum()

        # Output
        pd.set_option('display.float_format', '{:.4f}'.format)
        if debug:
            print()
            print(f"-----#####-----#####-----#####-----#####----- Naive Discrete Monitoring, n = {n} -----#####-----#####-----#####-----#####-----\n")
            print({"Naive Hits\n": naive_hits, "Naive Knock-Out Payoff\n": naive_ko_payoff,
                   "Naive Monte Carlo Payoff\n": naive_monte_carlo_payoff})

        monte_carlo_prices.iloc[k, 2] = naive_monte_carlo_payoff

    monte_carlo_prices["Differences"] = monte_carlo_prices["Bridge Correction Price"] - monte_carlo_prices["Naive Monitoring Price"]
    monte_carlo_prices["Naive Error"] = monte_carlo_prices["Naive Monitoring Price"] - monte_carlo_prices["Bridge Correction Price"].iloc[-1,]
    monte_carlo_prices["Bridge Error"] = monte_carlo_prices["Bridge Correction Price"] - monte_carlo_prices["Bridge Correction Price"].iloc[-1,]

    if plot:
        cols1 = ["Bridge Correction Price", "Naive Monitoring Price"]
        cols2 = ["Bridge Error", "Naive Error"]
        fig1 = delta_engine_plotter(plot, (12, 4), cols1, n_lengths[:-1], monte_carlo_prices.iloc[:-1,], 1, 2, x_label="n lengths")
        fig2 = delta_engine_plotter(plot, (12, 4), cols2, monte_carlo_prices["dt"].tolist()[:-1], monte_carlo_prices.iloc[:-1,], 1, 2, x_label="dt")
        return monte_carlo_prices, fig1, fig2
    else:
        return monte_carlo_prices


n_lens = np.arange(100, 10000, 100).tolist()
out = discretisation_bridge_monitoring(1000, 955, 0.2**0.5, 1, 0.1, n_lens, 10000, 1200, plot = True)

#output = discretisation_bridge_monitoring(1000, 955, 0.2**0.5, 1, 0.1, n_lens, 1000, 1200, 2.5, plot = True, debug = True)

