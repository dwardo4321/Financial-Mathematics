# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"C:\Users\Tapson\Downloads\Financial-Mathematics")))

# %%
from Stochastic_Core_Library.brownian_path_generator import path_generator
import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as stats
import pandas as pd

# %%
# 1 --------------------- Quadratic variation for a random walk --------------------------------------------------------
def qv_random_walk(n: int, loc: float, scale: float, plot=False):
    x_i = np.random.normal(loc, scale, size=n)
    m_i = []; qv_i = []
    m_i.append(x_i[0]); qv_i.append(x_i[0] ** 2)
    for i in range(len(x_i)):
        m_i.append(x_i[i] + m_i[i])
        qv_i.append((x_i[i] ** 2) + qv_i[i])
    out = pd.DataFrame({"Random Walk": m_i, "Quadratic Variation": qv_i})
    # ------------------------------------------------
    if plot:
        fig = plt.figure(figsize=(18, 5), facecolor="darkgrey")
        y_labels = ["M_k", "Cumulative Value"]
        titles = ["Symmetric Random Walk", "Quadratic Variation vs Theoretical Variance"]

        for i, data in enumerate(out.columns):
            sub = fig.add_subplot(1, 2, i + 1)
            plt.gca().set_facecolor("white")
            sub.plot(out.index, out[data])
            if data == "Quadratic Variation":
                sub.plot(out.index, (scale ** 2) * out.index, linestyle='--')
                sub.legend(["Quadratic Variation", "Theoretical Variance"], edgecolor='navy')
            sub.set_ylabel(y_labels[i], fontsize=10)
            sub.set_xlabel('k', fontsize=10)
            sub.set_title(titles[i], fontsize=12)
            plt.grid()
            plt.tight_layout()
        return out.head(), plt.show()
    else:
        return out.head()


# print(qv_random_walk(1000, 0, 0.1, plot=True))

# %%
# 2 --------------------- Quadratic variation of Brownian motion -------------------------------------------------------
def qv_brownian_motion(scale: float, t, n: int, num_gens: int, plot=False, correlated: tuple[bool, np.ndarray] = (False, None)):
    # ------------------------------------------------
    if correlated[0]:
        path = path_generator(n = n, t = t, num_gens = num_gens, correlated =(True, correlated[1]))
    else:
        path = path_generator(n = n, t = t, num_gens = num_gens)
    # ------------------------------------------------
    qv_bm = np.zeros((n-1, num_gens))
    for j in range(num_gens):
        qv_bm[:, j] = np.cumsum((np.diff(path.iloc[:, j]))**2, axis = 0)
    qv_bm = pd.DataFrame(qv_bm, columns=np.arange(1, num_gens+1, 1))
    # ------------------------------------------------
    if plot:
        dt = t/(n-1); x = np.arange(0, t, dt)
        plt.figure(figsize=(10, 5), facecolor='darkgrey')
        plt.gca().set_facecolor("white")
        plt.plot(x, qv_bm.values)
        plt.xlabel('t', fontsize=15)
        plt.ylabel('Quadratic Variation', fontsize=15)
        plt.grid()
        return qv_bm, plt.show()
    else:
        return qv_bm

# %%
print(qv_brownian_motion(0.1, 2, 1000, 1, plot=True))

# %%
# 3 --------------------- Realized volatility from GBM -----------------------------------------------------------------
# This one reduces to the QV of a BM
def geometric_bm(s_0: float, sd: float, t_n: float, rate: float, n_all: int, n_sims: int):
    bms = path_generator(n_all, t_n, n_sims)
    paths = pd.DataFrame()
    #col = range(1, n_sims + 1, 1)
    for i in range(n_sims):
        path_i = s_0 * np.exp(sd*bms.iloc[:, i].to_numpy() + (rate - 0.5*(sd**2))*np.linspace(0, t_n, n_all))
        paths = pd.concat([paths, pd.DataFrame({i+1 :path_i})], axis = 1)

    return paths

#test = geometric_bm(1000, 0.2, 20, 0.1, 1000, 5)
#print(np.log(test))
#test_1 = test.iloc[:,0].to_numpy()
#print(test_1[:, None][10, 0])
#plt.figure(figsize=(18, 5))
#plt.plot(test)
#plt.show()

# t = 20; n = 1000000; sig = 0.5; alph = 0.1
# s_t = geometric_bm(1000, sd = sig, t_n = t, ret = alph, n_all = n)
# log_ret = np.log(s_t.iloc[:-1, 0].to_numpy() / s_t.iloc[1:, 0].to_numpy())

# var_est = (1/t) * sum(log_ret ** 2)
# print(f"{var_est:.5f}")                     # Sum of the squared log returns over the period 0-T
# var = sig ** 2
# print(f"{var:.5f}")                       # Variance

# %%
# 4 --------------------- Quadratic covariation / cross-variation ------------------------------------------------------
from typing import Literal, Union
from dataclasses import dataclass

choice = Literal["Const", "Seasonal", "GBM"]
# --- parameter bundles ---
@dataclass
class ParamsConst:
    mu: float
    sig: float

@dataclass
class ParamsSeasonal:
    a_0: float; a_1: float
    b_0: float; b_1: float

@dataclass
class ParamsGBM:
    mu: float
    sig: float

def ito_process_gen(num_assets: int, assets_t0: list, vol_drift: choice, parameters: Union[ParamsConst, ParamsSeasonal, ParamsGBM],
                    n = 1000, t = 5, num_gens = 2, correlated: tuple[bool, np.ndarray] = (False, None)):

    if correlated[0]:
        bm = path_generator(n, t, num_gens, correlated=(True, correlated[1]))
    else:
        bm = path_generator(n, t, num_gens)

    prices = np.empty((n, num_assets))
    prices[0, :] = assets_t0
    time = np.linspace(0, t, n)
    dt = t/(n-1)
    dw = bm.diff().iloc[1:].to_numpy()
    eps = 1e-5

    for i in range(num_assets):
        for j in range(n-1):
            if vol_drift == "Const":
                drift = parameters.mu
                vol = parameters.sig
                prices[j + 1, i] = prices[j, i] + drift * dt + vol * dw[j, i]
            elif vol_drift == "Seasonal":
                drift_calc = parameters.a_0 + parameters.a_1 * np.sin(2 * np.pi * time[j])
                vol_calc = parameters.b_0 + parameters.b_1 * np.cos(2 * np.pi * time[j])
                drift = max(eps, float(drift_calc))
                vol = max(eps, float(vol_calc))
                prices[j + 1, i] = prices[j, i] + drift * dt + vol * dw[j, i]
            else:
                mu = parameters.mu  # requires ParamsGBM(mu, sig)
                sig = parameters.sig
                #s_t = prices[j, i]  # current price for asset i at step j
                drift = mu # * s_t  # θ(t,s_t)
                vol = sig
                prices[j + 1, i] = prices[j, i] * np.exp((drift - 0.5 * (vol ** 2)) * dt + vol * dw[j, i])

    pd.set_option('display.float_format', '{:.4f}'.format)
    prices = pd.DataFrame(prices, columns = np.arange(1, num_assets+1, 1))
    return prices

# %%
#asset_t0 = [1523, 2687, 742, 1492, 2531, 2741, 2480, 1603, 984]
#assets_no = 9

#para_gbm = ParamsGBM(mu = 0.1, sig = 0.15)
#S_t_1 = ito_process_gen(num_assets = assets_no, assets_t0 = asset_t0, vol_drift = "GBM", parameters = para_gbm, n = 100, t = 1, num_gens = assets_no)
#print(S_t_1.head())

#para_sns = ParamsSeasonal(a_0 = 0.05, a_1 = 0.1, b_0 = 0.06, b_1 = 0.11)
#S_t_2 = ito_process_gen(num_assets = assets_no, assets_t0 = asset_t0, vol_drift = "Seasonal", parameters = para_sns, n = 100, t = 1, num_gens = assets_no)
#print(S_t_2.head())

#para_const = ParamsConst(mu = 0.05, sig = 0.09)
#S_t_3 = ito_process_gen(num_assets = assets_no, assets_t0 = asset_t0, vol_drift = "Const", parameters = para_const, n = 100, t = 1, num_gens = assets_no)
#print(S_t_3.head())
