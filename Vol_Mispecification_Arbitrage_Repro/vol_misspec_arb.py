from Stochastic_Core_Library.quad_var_cov_estimator import geometric_bm
#from Delta_Hedging_Simulator.bsm_put_call_price import bsm_put_call_pricer
from Stochastic_Core_Library.brownian_path_generator import path_generator
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

def vol_misspec_pc_pricer(parameters, sig_real: float, bank_wthdrw: float, replic_error: float, plot: bool=False):
    s_0 = parameters[0]; strike = parameters[1]; sig = parameters[2]
    rate = parameters[3]; T = parameters[4]; alpha = parameters[5]
    n = parameters[6]; risk_neutral_pricing = parameters[7]

