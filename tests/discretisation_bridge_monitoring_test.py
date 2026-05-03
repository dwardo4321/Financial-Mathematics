#%%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\Users\Tapson\Downloads\Financial-Mathematics\src")))

from risk.discrete_monitoring_bias import brownian_bridge_monitoring

import numpy as np

#%%
n_lens = np.arange(100, 10000, 100).tolist()
bbm = brownian_bridge_monitoring(1000, 955, 0.2**0.5, 1, 0.1, n_lens, 10000, 1200)
out = bbm.discretisation_bridge_monitoring()

#%%
#output = discretisation_bridge_monitoring(1000, 955, 0.2**0.5, 1, 0.1, n_lens, 1000, 1200, 2.5, debug = True)
