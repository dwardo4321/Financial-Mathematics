import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

class utilityDGE:

    def array_def(self, n):

        c_tx = np.empty(n)
        p_tx = np.empty(n)
        delta_call = np.empty(n)
        delta_put = np.empty(n)
        bank_call = np.empty(n)
        bank_put = np.empty(n)
        portfolio_call = np.empty(n)
        portfolio_put = np.empty(n)

        return c_tx, p_tx, delta_call, delta_put, bank_call, bank_put, portfolio_call, portfolio_put


    def time_steps_gamma_delta_pricer(self, time, s_t, strike, rate, sig):

        if time == 0:
            c_tx = np.maximum(s_t - strike, 0)
            p_tx = np.maximum(strike - s_t, 0)
            gamma_call_put = 0
            if s_t > strike:
                delta_call = 1
                delta_put = 0
            elif s_t < strike:
                delta_call = 0
                delta_put = -1
            else:
                delta_call = 1e-5
                delta_put = 1e-5

        else:
            d1 = (np.log(s_t / strike) + (rate + 0.5 * (sig ** 2)) * time) / (sig * np.sqrt(time))
            d2 = d1 - sig * np.sqrt(time)
            c_tx = (s_t * stats.norm.cdf(d1)) - (strike * np.exp(-rate * time) * stats.norm.cdf(d2))
            p_tx = c_tx - s_t + (strike * np.exp(-rate * time))
            delta_call = stats.norm.cdf(d1)
            delta_put = -stats.norm.cdf(-d1)
            gamma_call_put = 1 / (sig * s_t * np.sqrt(time)) * (1 / np.sqrt(2 * np.pi) * np.exp((-d1 ** 2) / 2))

        return c_tx, p_tx, delta_call, delta_put, gamma_call_put

    def time_steps_0_portfolio(self, s_t, delta_call, delta_put, c_tx, p_tx):

        portfolio_call = c_tx
        bank_call = portfolio_call - delta_call * s_t
        portfolio_put = p_tx
        bank_put = portfolio_put - delta_put * s_t

        return portfolio_call, bank_call, portfolio_put, bank_put

    def time_steps_i_portfolio(self, i, bank_call, bank_put, rate, dt, s_t, delta_call, delta_put, c_tx, p_tx):

        bank_call = (bank_call[i - 1] * np.exp(rate * dt)) - ((delta_call[i] - delta_call[i - 1]) * s_t[i])
        bank_put = (bank_put[i - 1] * np.exp(rate * dt)) - ((delta_put[i] - delta_put[i - 1]) * s_t[i])

        portfolio_call = delta_call[i] * s_t[i] + bank_call
        portfolio_put = delta_put[i] * s_t[i] + bank_put

        return bank_call, bank_put, portfolio_call, portfolio_put

    def data_display(self, disp_width):
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", disp_width)
        pd.set_option("display.expand_frame_repr", False)
        return None
    

    def run_simultation(self, no_of_sims: int, unique: bool, function, parameters_1, parameters_2 = None):
        hedge_error_call = np.empty(no_of_sims)
        hedge_error_put = np.empty(no_of_sims)

        if unique:
            for i in range(no_of_sims):
                data_call = function(parameters_1)
                data_put = function(parameters_2)
                hedge_error_call[i] = data_call["Portfolio Call"].iloc[-1] - data_call["Call Price"].iloc[-1]
                hedge_error_put[i] = data_put["Portfolio Put"].iloc[-1] - data_put["Put Price"].iloc[-1]
        else:
            for i in range(no_of_sims):
                data = function(parameters_1)
                hedge_error_call[i] = data["Portfolio Call"].iloc[-1] - data["Call Price"].iloc[-1]
                hedge_error_put[i] = data["Portfolio Put"].iloc[-1] - data["Put Price"].iloc[-1]

        return hedge_error_call, hedge_error_put
    

    def delta_engine_plotter(self, dim, cols, t_0n, out, nrow, ncol, x_label: str = "Time"):
        fig = plt.figure(figsize=dim, facecolor='darkgrey')

        for i, data in enumerate(cols):
            sub = fig.add_subplot(nrow, ncol, i + 1)
            sub.plot(t_0n, out[data].values)
            sub.set_xlabel(x_label, fontsize=12)
            sub.set_ylabel(data, fontsize=12)
            sub.grid()
        fig.tight_layout()
        plt.show(block=False)
        return fig
    