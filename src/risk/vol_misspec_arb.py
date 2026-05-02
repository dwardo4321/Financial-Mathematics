# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\Users\Tapson\Downloads\Financial-Mathematics\src")))

from simulation.pricing_paths import pricePATH
from utilities.utilities_hedging.utilities_delta_gamma_engine import utilityDGE
from plotting.repo_plots import engine_plotter

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

#%%
class volatility_misspec_arbitrage:

    def __init__(self, s_0: float, strike: float, vol: float, rate: float, T: float,
                  alpha: float, n: int, risk_neutral_pricing: float, vol_real: float):
        
        self.s_0 = s_0 
        self.strike = strike 
        self.vol = vol 
        self.rate = rate 
        self.T = T 
        self.alpha = alpha 
        self.n = n 
        self.risk_neutral_pricing = risk_neutral_pricing 
        self.vol_real = vol_real

    def volatility_misspecification_pricer(self):

        gbm = pricePATH()
        utils = utilityDGE()

        drift = self.rate if self.risk_neutral_pricing else self.alpha
        s_t = gbm.geometric_bm(self.s_0, self.vol_real, self.T, drift, self.n, n_sims=1)["Price"].to_numpy()

        c_tx, p_tx, delta_call, delta_put, bank_call, bank_put, portfolio_call, portfolio_put = utils.array_def(self.n)
        gamma_call_put = np.zeros(self.n)
        bank_withdrawal = np.zeros(self.n)
        cumulative_withdrawal = np.zeros(self.n)
        replication_error_call = np.zeros(self.n)
        replication_error_put = np.zeros(self.n)

        self.t_0n = np.linspace(0, self.T, self.n)
        tau = self.T - self.t_0n
        dt = self.T / (self.n-1)

        for i, time in enumerate(tau):
            c_tx[i], p_tx[i], delta_call[i], delta_put[i], gamma_call_put[i] = utils.time_steps_gamma_delta_pricer(time, s_t[i], self.strike, self.rate, self.vol)

            if i == 0:
                portfolio_call[0], bank_call[0], portfolio_put[0], bank_put[0] = utils.time_steps_0_portfolio(s_t[0], delta_call[0], delta_put[0], c_tx[0], p_tx[0])

            else:
                bank_withdrawal[i] = 0.5 * (self.vol_real ** 2 - self.vol ** 2) * (s_t[i-1]**2) * gamma_call_put[i-1] * dt
                cumulative_withdrawal[i] = cumulative_withdrawal[i - 1] + bank_withdrawal[i]

                bank_call_i, bank_put_i, _, _ = utils.time_steps_i_portfolio(i, bank_call, bank_put, self.rate, dt, s_t, delta_call, delta_put, c_tx, p_tx)

                bank_call[i] = bank_call_i - bank_withdrawal[i]
                bank_put[i] = bank_put_i - bank_withdrawal[i]

                portfolio_call[i] = delta_call[i] * s_t[i] + bank_call[i]
                portfolio_put[i] = delta_put[i] * s_t[i] + bank_put[i]

                replication_error_call[i] = portfolio_call[i] - c_tx[i]
                replication_error_put[i] = portfolio_put[i] - p_tx[i]

        pd.set_option('display.float_format', '{:.4f}'.format)
        vmp = pd.DataFrame({"Time": self.t_0n, "Call Price": c_tx, "Put Price": p_tx, "Asset Prices": s_t, "Delta Call": delta_call,
                            "Delta Put": delta_put, "Bank Call": bank_call, "Portfolio Call": portfolio_call, "Bank Put": bank_put,
                            "Portfolio Put": portfolio_put, "Gamma": gamma_call_put, "Withdrawal": bank_withdrawal,
                            "Cumulative Withdrawal": cumulative_withdrawal, "Replication Error Call": replication_error_call,
                            "Replication Error Put": replication_error_put,}).set_index("Time")

        self.cols = ["Call Price", "Put Price", "Asset Prices", "Gamma", "Withdrawal",
                "Cumulative Withdrawal", "Replication Error Call", "Replication Error Put"]

        self.vmp = vmp

        utils.data_display(1000)
        return self.vmp


    def vmp_plot(self):
            
            if not hasattr(self, "vmp"):
                self.volatility_misspecification_pricer()
                
            eplt = engine_plotter()        
            fig = eplt.delta_engine_plotter((20, 12.8), self.cols, self.t_0n, self.vmp, nrow = 4, ncol = 4)

# %%
