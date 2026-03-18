from Stochastic_Core_Library.quad_var_cov_estimator import geometric_bm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

def bsm_put_call_pricer(s_0: float, strike: float, sig: float, rate: float, T: int, alpha: float, n: int, plot: bool = False,risk_neutral_pricing: bool = False):

    drift = rate if risk_neutral_pricing else alpha
    s_t = geometric_bm(s_0, sig, T, drift, n)["Price"].to_numpy()

    dt = T/(n-1)
    t_0n = np.linspace(0, T, n)
    tau = T - t_0n
    c_tx = np.empty(n)
    p_tx = np.empty(n)
    delta_call = np.empty(n)
    delta_put = np.empty(n)
    bank_call = np.empty(n)
    bank_put = np.empty(n)
    portfolio_call = np.empty(n)
    portfolio_put = np.empty(n)

    for i, time in enumerate(tau):
        if time == 0:
            c_tx[i] = np.maximum(s_t[i] - strike, 0)
            p_tx[i] = np.maximum(strike - s_t[i], 0)
            if s_t[i] > strike:
                delta_call[i] = 1
                delta_put[i] = 0
            elif s_t[i] < strike:
                delta_call[i] = 0
                delta_put[i] = -1
            else:
                delta_call[i] = 1e-5
                delta_put[i] = 1e-5

        else:
            d1 = (np.log(s_t[i] / strike) + (rate + 0.5 * (sig**2)) * time) / (sig*np.sqrt(time))
            d2 = d1 - sig * np.sqrt(time)
            c_tx[i] = (s_t[i] * stats.norm.cdf(d1)) - (strike * np.exp(-rate * time) * stats.norm.cdf(d2))
            p_tx[i] = c_tx[i] - s_t[i] + (strike * np.exp(-rate * time))
            delta_call[i] = stats.norm.cdf(d1)
            delta_put[i] = -stats.norm.cdf(-d1)

        if i == 0:
            portfolio_call[0] = c_tx[0]
            bank_call[0] = portfolio_call[0] - delta_call[0] * s_t[0]
            portfolio_put[0] = p_tx[0]
            bank_put[0] = portfolio_put[0] - delta_put[0] * s_t[0]

        else:
            bank_call[i] = (bank_call[i-1] * np.exp(rate * dt)) - ((delta_call[i] - delta_call[i-1]) * s_t[i])
            portfolio_call[i] = delta_call[i] * s_t[i] + bank_call[i]
            bank_put[i] = (bank_put[i-1] * np.exp(rate * dt)) - ((delta_put[i] - delta_put[i-1]) * s_t[i])
            portfolio_put[i] = delta_put[i] * s_t[i] + bank_put[i]

    pd.set_option('display.float_format', '{:.4f}'.format)
    out = pd.DataFrame({"Time": t_0n, "Call Price": c_tx, "Put Price": p_tx, "Asset Prices": s_t, "Delta Call": delta_call,
                        "Delta Put":delta_put,"Bank Call": bank_call, "Portfolio Call": portfolio_call, "Bank Put": bank_put,
                        "Portfolio Put": portfolio_put}).set_index("Time")
    if plot:
        fig = plt.figure(figsize=(11, 8.5), facecolor='darkgrey')
        cols = ["Call Price", "Put Price", "Asset Prices"]
        for i, data in enumerate(cols):
            sub = fig.add_subplot(3, 1, i+1)
            sub.plot(t_0n, out[data].values)
            sub.set_xlabel("Time", fontsize=12)
            sub.set_ylabel(data, fontsize=12)
            sub.grid()
        fig.tight_layout()
        return out, plt.show()
    else:
        return out

#delta_engine = bsm_put_call_pricer(s_0 = 1000, strike = 985, sig = 0.2, rate = 0.05, T = 10, alpha = 0.05, n = 1000, risk_neutral_pricing = False, plot = True)
#pd.set_option("display.max_columns", None)
#pd.set_option("display.width", 1000)
#pd.set_option("display.expand_frame_repr", False)
#print(delta_engine)

def hedging_error_dist(parameters, no_of_sims: int):

    s_0 = parameters[0]; strike = parameters[1]; sig = parameters[2]; rate = parameters[3]; T = parameters[4]; alpha = parameters[5]
    n = parameters[6]; risk_neutral_pricing = parameters[7]; plot = parameters[8]

    hedge_error_call = np.empty(no_of_sims)
    hedge_error_put = np.empty(no_of_sims)

    for i in range(no_of_sims):
        data = bsm_put_call_pricer(s_0, strike, sig, rate, T, alpha, n, risk_neutral_pricing, plot)
        hedge_error_call[i] = data["Portfolio Call"].iloc[-1] - data["Call Price"].iloc[-1]
        hedge_error_put[i] = data["Portfolio Put"].iloc[-1] - data["Put Price"].iloc[-1]

    df = pd.DataFrame({"Sim": np.arange(no_of_sims),"Hedging Call Error": hedge_error_call, "Hedging Put Error": hedge_error_put}).set_index("Sim")
    return df

pars = [1000, 985, 0.2, 0.05, 10, 0.05, 1000, False, False]
he = hedging_error_dist(pars, no_of_sims = 1000)

plt.figure(figsize = (10, 10))
plt.hist(he["Hedging Call Error"])
plt.show()


