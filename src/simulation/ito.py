import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"C:\Users\Tapson\Downloads\Financial-Mathematics")))
from simulation.pricing_paths import path_generator
import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as stats
import pandas as pd

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