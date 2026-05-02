# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"C:\Users\Tapson\Downloads\Financial-Mathematics")))

from simulation.pricing_paths import pricePATH
from plotting.repo_plots import engine_plotter

import numpy as np
import pandas as pd
import scipy.stats as stats

#%%
class brownian_bridge_monitoring:

    def __init__(self, s_0: float, strike: float, sd: float, T: float, rate: float, n_lengths: list, n_sims: int, barrier_raw: float, debug: bool = False):
        
        self.s_0 = s_0
        self.strike = strike
        self.sd = sd
        self.T = T
        self.rate = rate
        self.n_lengths = n_lengths
        self.n_sims = n_sims
        self.barrier_raw = barrier_raw
        self.debug = debug

    def discretisation_bridge_monitoring(self):

        self.n_lengths = list(self.n_lengths)
        self.n_lengths.append(self.n_lengths[-1] + 10000)  # for benchmark price

        monte_carlo_prices = pd.DataFrame(columns = ["dt", "Bridge Correction Price", "Naive Monitoring Price"], index = self.n_lengths)

        gbm = pricePATH()
        for k, n in enumerate(monte_carlo_prices.index):
            prob = np.zeros((n-1, self.n_sims))
            dt = self.T / (n-1)
            monte_carlo_prices.iloc[k, 0] = dt

            barrier = np.log(self.barrier_raw)
            gbm_prices = gbm.geometric_bm(self.s_0, self.sd, self.T, self.rate, n, self.n_sims)

        # Bridge Correction Monitoring -------------------------------------------------------------------------------------
            # Simulated Log Prices
            simulation = np.log(gbm_prices.to_numpy())

            # Hit Probabilities
            for i in range(self.n_sims):
                for j in range(n-1):
                    if simulation[j,i] < barrier and simulation[j+1,i] < barrier:
                        prob[j, i] = np.exp(-(2*(barrier - simulation[j,i]) * (barrier - simulation[j+1,i])) / (self.sd**2 * dt))
                    else:
                        prob[j, i] = 1

            prob = pd.DataFrame(prob, columns=np.arange(1, self.n_sims + 1, 1))

            # Bernoulli Path Generator
            rand_uniform = stats.uniform.rvs(0, 1, np.shape(prob))
            path_simulations = (prob > rand_uniform).any(axis=0).astype(int)

            # Path Survival Probabilities
            path_survival_prob = (1 - prob).prod(axis=0)

            # Survival Weighted Payoff
            maturity_survival_weighted_payoff = path_survival_prob * np.maximum(gbm_prices.iloc[-1, :] - self.strike, 0)

            # Monte Carlo Payoff
            monte_carlo_bridge_corrected_payoff = np.exp(-self.rate * self.T) * (1 / self.n_sims) * maturity_survival_weighted_payoff.sum()

            # Output
            if self.debug:
                pd.set_option('display.float_format', '{:.4f}'.format)
                print()
                print(f"-----#####-----#####-----#####-----#####----- Bridge Correction Monitoring, n = {n} -----#####-----#####-----#####-----#####-----\n")
                print({"Log Price Simulations\n": simulation, "Hit Probabilities\n": prob, "Bernoulli Path Simulations\n": path_simulations,
                    "Path Survival Probabilities\n": path_survival_prob, "Survival Weighted Payoff\n": maturity_survival_weighted_payoff,
                    "Monte Carlo Bridge Corrected Payoff\n": monte_carlo_bridge_corrected_payoff})

            monte_carlo_prices.iloc[k, 1] = monte_carlo_bridge_corrected_payoff

        # Naive Discrete Monitoring ----------------------------------------------------------------------------------------
            # Naive Hits
            naive_hits = (gbm_prices >= self.barrier_raw).any(axis=0).astype(int)

            # Naive Knock-Out Payoff
            naive_ko_payoff = (1-naive_hits) * np.maximum(gbm_prices.iloc[-1, :] - self.strike, 0)

            # Naive Monte Carlo Payoff
            naive_monte_carlo_payoff = np.exp(-self.rate * self.T) * (1 / self.n_sims) * naive_ko_payoff.sum()

            # Output
            pd.set_option('display.float_format', '{:.4f}'.format)
            if self.debug:
                print()
                print(f"-----#####-----#####-----#####-----#####----- Naive Discrete Monitoring, n = {n} -----#####-----#####-----#####-----#####-----\n")
                print({"Naive Hits\n": naive_hits, "Naive Knock-Out Payoff\n": naive_ko_payoff,
                    "Naive Monte Carlo Payoff\n": naive_monte_carlo_payoff})

            monte_carlo_prices.iloc[k, 2] = naive_monte_carlo_payoff

        monte_carlo_prices["Differences"] = monte_carlo_prices["Bridge Correction Price"] - monte_carlo_prices["Naive Monitoring Price"]
        monte_carlo_prices["Naive Error"] = monte_carlo_prices["Naive Monitoring Price"] - monte_carlo_prices["Bridge Correction Price"].iloc[-1,]
        monte_carlo_prices["Bridge Error"] = monte_carlo_prices["Bridge Correction Price"] - monte_carlo_prices["Bridge Correction Price"].iloc[-1,]

        self.monte_carlo_prices = monte_carlo_prices

        return self.monte_carlo_prices

    # --------------------------------------------------------------------------------------

    def dbm_plot(self):

        if not hasattr(self, "self.monte_carlo_prices"):
            self.discretisation_bridge_monitoring()

        eplt = engine_plotter()

        cols1 = ["Bridge Correction Price", "Naive Monitoring Price"]
        cols2 = ["Bridge Error", "Naive Error"]
        self.fig1 = eplt.delta_engine_plotter((12, 4), cols1, self.monte_carlo_prices.index[:-1], self.monte_carlo_prices.iloc[:-1,], 1, 2, x_label="n lengths")
        self.fig2 = eplt.delta_engine_plotter((12, 4), cols2, self.monte_carlo_prices["dt"].tolist()[:-1], self.monte_carlo_prices.iloc[:-1,], 1, 2, x_label="dt")
        
        return self.fig1, self.fig2
