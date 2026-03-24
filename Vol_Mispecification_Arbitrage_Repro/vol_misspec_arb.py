from Stochastic_Core_Library.quad_var_cov_estimator import geometric_bm
from Utility_Functions.utility import array_def, time_steps_gamma_delta_pricer, time_steps_0_portfolio, time_steps_i_portfolio, delta_engine_plotter, data_display
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

def volatility_misspecification_pricer(parameters, sig_real: float, plot: bool=False):
    s_0 = parameters[0]; strike = parameters[1]; sig = parameters[2]
    rate = parameters[3]; T = parameters[4]; alpha = parameters[5]
    n = parameters[6]; risk_neutral_pricing = parameters[7]

    drift = rate
    s_t = geometric_bm(s_0, sig_real, T, drift, n)["Price"].to_numpy()

    c_tx, p_tx, delta_call, delta_put, bank_call, bank_put, portfolio_put, portfolio_call = array_def(n)
    gamma_call_put = np.zeros(n)
    bank_withdrawal = np.zeros(n)
    cumulative_withdrawal = np.zeros(n)
    replication_error_call = np.zeros(n)
    replication_error_put = np.zeros(n)

    t_0n = np.linspace(0, T, n)
    tau = T - t_0n
    dt = T/(n-1)

    for i, time in enumerate(tau):
        c_tx[i], p_tx[i], delta_call[i], delta_put[i], gamma_call_put[i] = time_steps_gamma_delta_pricer(time, s_t[i], strike, rate, sig)

        if i == 0:
            portfolio_call[0], bank_call[0], portfolio_put[0], bank_put[0] = time_steps_0_portfolio(s_t[0], delta_call[0], delta_put[0], c_tx[0], p_tx[0])

        else:
            bank_withdrawal[i] = 0.5 * (sig_real ** 2 - sig ** 2) * (s_t[i-1]**2) * gamma_call_put[i-1] * dt
            cumulative_withdrawal[i] = cumulative_withdrawal[i - 1] + bank_withdrawal[i]

            bank_call[i], bank_put[i], portfolio_call[i], portfolio_put[i] = time_steps_i_portfolio(i, bank_call, bank_put, rate, dt, s_t, delta_call, delta_put, c_tx, p_tx) + bank_withdrawal[i]

            replication_error_call[i] = portfolio_call[i] - c_tx[i]
            replication_error_put[i] = portfolio_put[i] - p_tx[i]

    pd.set_option('display.float_format', '{:.4f}'.format)
    out = pd.DataFrame({"Time": t_0n, "Call Price": c_tx, "Put Price": p_tx, "Asset Prices": s_t, "Delta Call": delta_call,
                        "Delta Put": delta_put, "Bank Call": bank_call, "Portfolio Call": portfolio_call, "Bank Put": bank_put,
                        "Portfolio Put": portfolio_put, "Gamma": gamma_call_put, "Withdrawal": bank_withdrawal,
                        "Cumulative Withdrawal": cumulative_withdrawal, "Replication Error Call": replication_error_call,
                        "Replication Error Put": replication_error_put,}).set_index("Time")

    cols = ["Call Price", "Put Price", "Asset Prices", "Gamma", "Withdrawal",
            "Cumulative Withdrawal", "Replication Error Call", "Replication Error Put"]

    if plot:
        fig = delta_engine_plotter(plot, (20, 12.8), cols, t_0n, out, nrow = 4, ncol = 4)
        data_display(1000)
        return out, fig
    else:
        data_display(1000)
        return out


pars_c = [1000, 1005, 0.2, 0.05, 2, 0.05, 100000, True]
vol_mis = volatility_misspecification_pricer(pars_c, 0.25, True)
print(vol_mis)