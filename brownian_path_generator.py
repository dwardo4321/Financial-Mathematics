import numpy as np
import matplotlib.pyplot as plt

N = 100000
norm_rv1 = np.random.normal(0, 2, N-1)
norm_rv2 = np.random.normal(0, 2, N-1)

#  1 ----------------------------------------------------------
W_T1 = np.zeros((N, 1)); W_T2 = np.zeros((N, 1))

norm_rv = [norm_rv1, norm_rv2]
wp = [W_T1, W_T2]

for data, w_t in zip(norm_rv, wp):
    for i, rv in enumerate(data):
        w_t[i + 1, ] += w_t[i, ] + rv

#  2 ----------------------------------------------------------
# Shorter, no loops
# W_T1 = np.concatenate(([0], np.cumsum(norm_rv1)))[:, None]
# W_T2 = np.concatenate(([0], np.cumsum(norm_rv2)))[:, None]

phi = 0.95
W_T3 = phi * W_T1 + np.sqrt(1 - phi**2) * W_T2

plt.figure(figsize = (19, 10))
plt.plot(W_T1, color = 'b')
plt.plot(W_T2, color = 'r')
plt.plot(W_T3, color = 'black')
plt.show()
