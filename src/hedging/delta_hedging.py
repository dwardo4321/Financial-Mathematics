#%%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\Users\Tapson\Downloads\Financial-Mathematics\src")))

from utilities.utilities_hedging.utilities_delta_gamma_engine import utilityDGE
from simulation.pricing_paths import pricePATH

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#%%
class delta_hedge_engine:

    def __init__(self, s_0: float, strike: float, sig: float, rate: float, T: int, alpha: float, n: int, risk_neutral_pricing: bool = False):

        self.s_0 = s_0
        self.strike = strike
        self.sig = sig
        self.rate = rate
        self.T = T
        self.alpha = alpha
        self.n = n
        self.risk_neutral_pricing = risk_neutral_pricing

    # ---------------------------------------------------------------------------------------------------
    
    def bsm_put_call_pricer(self, parameters = None):

        if parameters is None:
            s_0 = self.s_0
            strike = self.strike
            sig = self.sig
            rate = self.rate
            T = self.T
            alpha = self.alpha
            n = self.n
            risk_neutral_pricing = self.risk_neutral_pricing
        else:
            s_0, strike, sig, rate, T, alpha, n, risk_neutral_pricing = parameters

        drift = rate if risk_neutral_pricing else alpha
        
        pp = pricePATH()
        s_t = pp.geometric_bm(s_0, sig, T, drift, n, 1).iloc[:,0].to_numpy()

        dt = T/(n-1)
        t_0n = np.linspace(0, T, n)
        tau = T - t_0n

        utils = utilityDGE()
        c_tx, p_tx, delta_call, delta_put, bank_call, bank_put, portfolio_call, portfolio_put = utils.array_def(n)

        for i, time in enumerate(tau):
            c_tx[i], p_tx[i], delta_call[i], delta_put[i], _ = utils.time_steps_gamma_delta_pricer(time, s_t[i], strike, rate, sig)

            if i == 0:
                portfolio_call[0], bank_call[0], portfolio_put[0], bank_put[0] = utils.time_steps_0_portfolio(s_t[0], delta_call[0], delta_put[0], c_tx[0], p_tx[0])

            else:
                bank_call[i], bank_put[i], portfolio_call[i], portfolio_put[i] = utils.time_steps_i_portfolio(i, bank_call, bank_put, rate, dt, s_t, delta_call, delta_put, c_tx, p_tx)


        pd.set_option('display.float_format', '{:.4f}'.format)
        bsm_cp = pd.DataFrame({"Time": t_0n, "Call Price": c_tx, "Put Price": p_tx, "Asset Prices": s_t, "Delta Call": delta_call,
                            "Delta Put":delta_put,"Bank Call": bank_call, "Portfolio Call": portfolio_call, "Bank Put": bank_put,
                            "Portfolio Put": portfolio_put}).set_index("Time")

        self.bsm_cp = bsm_cp
        
        cols = ["Call Price", "Put Price", "Asset Prices"]
   
        utils.data_display(1000)
        
        return self.bsm_cp
    
    # ---------------------------------------------------------------------------------------------------

    def plot_bsm_pc(self):

        if not hasattr(self, "bsm_cp"):
            self.bsm_put_call_pricer()  

        utils = utilityDGE()
        cols = ["Call Price", "Put Price", "Asset Prices"]
        fig = utils.delta_engine_plotter((11, 8.5), cols, self.bsm_cp.index, self.bsm_cp, nrow = 3, ncol = 1)

        return fig
    
    # ---------------------------------------------------------------------------------------------------

    def hedging_error_dist(self, no_of_sims: int = 5, parameters_call=None, parameters_put=None):

        utils = utilityDGE()

        if parameters_call and parameters_put:
            
            s_0_call = parameters_call[0]; strike_call = parameters_call[1]; sig_call = parameters_call[2]; rate_call = parameters_call[3]
            T_call = parameters_call[4]; alpha_call = parameters_call[5]; n_call = parameters_call[6]; risk_neutral_pricing_call = parameters_call[7]
            s_0_put = parameters_put[0]; strike_put = parameters_put[1]; sig_put = parameters_put[2]; rate_put = parameters_put[3]
            T_put = parameters_put[4]; alpha_put = parameters_put[5]; n_put = parameters_put[6]; risk_neutral_pricing_put = parameters_put[7]

            sims = np.linspace(0, no_of_sims, no_of_sims)

            hedge_error_call, hedge_error_put = utils.run_simultation(no_of_sims, True, self.bsm_put_call_pricer, parameters_call, parameters_put)
        
        elif parameters_call or parameters_put:

            parameters = parameters_call or parameters_put

            s_0 = parameters[0]; strike = parameters[1]; sig = parameters[2]; rate = parameters[3]; T = parameters[4]
            alpha = parameters[5]; n = parameters[6]; risk_neutral_pricing = parameters[7]

            hedge_error_call, hedge_error_put = utils.run_simultation(no_of_sims, False, self.bsm_put_call_pricer, parameters)

        else:

            parameters = [self.s_0, self.strike, self.sig, self.rate, self.T, self.alpha, self.n, self.risk_neutral_pricing]

            hedge_error_call, hedge_error_put = utils.run_simultation(no_of_sims, False, self.bsm_put_call_pricer, parameters)

        
        df = pd.DataFrame({"Sim": np.arange(no_of_sims),"Hedging Call Error": hedge_error_call, "Hedging Put Error": hedge_error_put}).set_index("Sim")
        results = {"Mean hedging call error": df["Hedging Call Error"].mean().round(4),
                "S.D hedging call error": stats.tstd(df["Hedging Call Error"]).round(4),
                "Mean hedging put error": df["Hedging Put Error"].mean().round(4),
                "S.D hedging put error": stats.tstd(df["Hedging Put Error"]).round(4)}
        
        self.df = df; self.results = results
        
        return self.df, self.results
    
    # ---------------------------------------------------------------------------------------------------

    def plot_hed(self):

        if not hasattr(self, "df"):
            self.hedging_error_dist()

        fig = plt.figure(figsize=(11, 8.5), facecolor='darkgrey')
        colum = ["Hedging Call Error", "Hedging Put Error"]

        for j, data in enumerate(colum):
            sub = fig.add_subplot(2, 1, j+1)
            #sub.plot(t_0n, df[data].values)
            sub.hist(self.df[data], bins=20)
            sub.set_xlabel("Error", fontsize=12)
            sub.set_ylabel(data, fontsize=12)
            sub.grid()

        fig.tight_layout()
        plt.show()

     # ---------------------------------------------------------------------------------------------------

# %%
