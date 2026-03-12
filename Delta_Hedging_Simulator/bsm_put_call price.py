from Stochastic_Core_Library.quad_var_cov_estimator import  geometric_bm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

def bsm_put_call_pricer(s_0: float, strike: float, sig: float, rate: float, T: int, alpha: float, n: int, plot: bool = False,risk_neutral_pricing: bool = False):

    drift = rate if risk_neutral_pricing else alpha
    s_t = geometric_bm(s_0, sig, T, drift, n)["Price"].to_numpy()

    t_0n = np.linspace(0, T, n)
    tau = T - t_0n
    c_tx = np.empty(n)
    p_tx = np.empty(n)

    for i, time in enumerate(tau):
        if time == 0:
            c_tx[i] = np.maximum(s_t[i] - strike, 0)
            p_tx[i] = np.maximum(strike - s_t[i], 0)
        else:
            d1 = (np.log(s_t[i] / strike) + (rate + 0.5 * (sig**2)) * time) / (sig*np.sqrt(time))
            d2 = d1 - sig * np.sqrt(time)
            c_tx[i] = (s_t[i] * stats.norm.cdf(d1)) - (strike * np.exp(-rate * time) * stats.norm.cdf(d2))
            p_tx[i] = c_tx[i] - s_t[i] + (strike * np.exp(-rate * time))

    pd.set_option('display.float_format', '{:.4f}'.format)
    out = pd.DataFrame({"Time": t_0n, "Call Price": c_tx, "Put Price": p_tx, "Asset Prices": s_t}).set_index("Time")
    if plot:
        fig = plt.figure(figsize=(14, 8.5), facecolor='darkgrey')
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

print(bsm_put_call_pricer(s_0 = 1000, strike = 985, sig = 0.2, rate = 0.05, T = 2, alpha = 0.05, n = 1000, risk_neutral_pricing = False, plot = True))

