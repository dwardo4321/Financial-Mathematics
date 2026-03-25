from Stochastic_Core_Library.quad_var_cov_estimator import geometric_bm
from Utility_Functions.utilities_delta_gamma_engine import array_def, time_steps_gamma_delta_pricer, time_steps_0_portfolio, time_steps_i_portfolio, delta_engine_plotter, data_display
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

def bsm_put_call_pricer(s_0: float, strike: float, sig: float, rate: float, T: int, alpha: float, n: int, risk_neutral_pricing: bool = False, plot: bool = False):

    drift = rate if risk_neutral_pricing else alpha
    s_t = geometric_bm(s_0, sig, T, drift, n)["Price"].to_numpy()

    dt = T/(n-1)
    t_0n = np.linspace(0, T, n)
    tau = T - t_0n
    c_tx, p_tx, delta_call, delta_put, bank_call, bank_put, portfolio_put, portfolio_call = array_def(n)

    for i, time in enumerate(tau):
        c_tx[i], p_tx[i], delta_call[i], delta_put[i], _ = time_steps_gamma_delta_pricer(time, s_t[i], strike, rate, sig)

        if i == 0:
            portfolio_call[0], bank_call[0], portfolio_put[0], bank_put[0] = time_steps_0_portfolio(s_t[0], delta_call[0], delta_put[0], c_tx[0], p_tx[0])

        else:
            bank_call[i], bank_put[i], portfolio_call[i], portfolio_put[i] = time_steps_i_portfolio(i, bank_call, bank_put, rate, dt, s_t, delta_call, delta_put, c_tx, p_tx)


    pd.set_option('display.float_format', '{:.4f}'.format)
    out = pd.DataFrame({"Time": t_0n, "Call Price": c_tx, "Put Price": p_tx, "Asset Prices": s_t, "Delta Call": delta_call,
                        "Delta Put":delta_put,"Bank Call": bank_call, "Portfolio Call": portfolio_call, "Bank Put": bank_put,
                        "Portfolio Put": portfolio_put}).set_index("Time")

    cols = ["Call Price", "Put Price", "Asset Prices"]

    if plot:
        fig = delta_engine_plotter(plot, (11, 8.5), cols, t_0n, out, nrow = 3, ncol = 1)
        data_display(1000)
        return out, fig
    else:
        data_display(1000)
        return out

delta_engine = bsm_put_call_pricer(s_0 = 980, strike = 985, sig = 0.1, rate = 0.09, T = 3, alpha = 0.05, n = 10000, risk_neutral_pricing = False, plot = True)
print(delta_engine)

def hedging_error_dist(no_of_sims: int, parameters_call=None, parameters_put=None, parameters=None, plot: bool = False, c_p_same_paras: bool = False):

    if c_p_same_paras:
        s_0 = parameters[0]; strike = parameters[1]; sig = parameters[2]; rate = parameters[3]; T = parameters[4]
        alpha = parameters[5]; n = parameters[6]; risk_neutral_pricing = parameters[7]

        hedge_error_call = np.empty(no_of_sims)
        hedge_error_put = np.empty(no_of_sims)

        for i in range(no_of_sims):
            data = bsm_put_call_pricer(s_0, strike, sig, rate, T, alpha, n, risk_neutral_pricing, plot = False)
            hedge_error_call[i] = data["Portfolio Call"].iloc[-1] - data["Call Price"].iloc[-1]
            hedge_error_put[i] = data["Portfolio Put"].iloc[-1] - data["Put Price"].iloc[-1]

    else:
        s_0_call = parameters_call[0]; strike_call = parameters_call[1]; sig_call = parameters_call[2]; rate_call = parameters_call[3]
        T_call = parameters_call[4]; alpha_call = parameters_call[5]; n_call = parameters_call[6]; risk_neutral_pricing_call = parameters_call[7]
        s_0_put = parameters_put[0]; strike_put = parameters_put[1]; sig_put = parameters_put[2]; rate_put = parameters_put[3]
        T_put = parameters_put[4]; alpha_put = parameters_put[5]; n_put = parameters_put[6]; risk_neutral_pricing_put = parameters_put[7]

        sims = np.linspace(0, no_of_sims, no_of_sims)
        hedge_error_call = np.empty(no_of_sims)
        hedge_error_put = np.empty(no_of_sims)

        for i in range(no_of_sims):
            data_call = bsm_put_call_pricer(s_0_call, strike_call, sig_call, rate_call, T_call, alpha_call, n_call, risk_neutral_pricing_call, plot = False)
            data_put = bsm_put_call_pricer(s_0_put, strike_put, sig_put, rate_put, T_put, alpha_put, n_put, risk_neutral_pricing_put, plot = False)
            hedge_error_call[i] = data_call["Portfolio Call"].iloc[-1] - data_call["Call Price"].iloc[-1]
            hedge_error_put[i] = data_put["Portfolio Put"].iloc[-1] - data_put["Put Price"].iloc[-1]

    df = pd.DataFrame({"Sim": np.arange(no_of_sims),"Hedging Call Error": hedge_error_call, "Hedging Put Error": hedge_error_put}).set_index("Sim")
    results = {"Mean hedging call error": df["Hedging Call Error"].mean().round(4),
               "S.D hedging call error": stats.tstd(df["Hedging Call Error"]).round(4),
               "Mean hedging put error": df["Hedging Put Error"].mean().round(4),
               "S.D hedging put error": stats.tstd(df["Hedging Put Error"]).round(4)}

    if plot:
        fig = plt.figure(figsize=(11, 8.5), facecolor='darkgrey')
        colum = ["Hedging Call Error", "Hedging Put Error"]
        for j, data in enumerate(colum):
            sub = fig.add_subplot(2, 1, j+1)
            #sub.plot(t_0n, df[data].values)
            sub.hist(df[data], bins=20)
            sub.set_xlabel("Error", fontsize=12)
            sub.set_ylabel(data, fontsize=12)
            sub.grid()
        fig.tight_layout()
        plt.show()

    return df, results

#pars_c = [1000, 985, 0.2, 0.05, 10, 0.05, 1000, False]
#pars_p = [945, 800, 0.15, 0.1, 15, 0.088, 1000, False]
#errors, summary = hedging_error_dist(parameters_call=pars_c, parameters_put=pars_p, no_of_sims = 100, plot=True)
#print(errors, "\n")
#print(summary)


