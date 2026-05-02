# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
class brownian_motion:

    def __init__(self, n: int, t: float, num_gens: int, correlation_matrix: np.array = None):
        self.n = n
        self.t = t
        self.num_gens = num_gens
        self.correlation_matrix = correlation_matrix
    
    # ---------------------------------------------------------------------------------------------------

    def simulate(self):
        dt = self.t / (self.n - 1)

        if self.correlation_matrix is not None:
            l_matrix = np.linalg.cholesky(self.correlation_matrix)
            z = np.random.normal(0, 1, size=(self.n - 1, self.num_gens))
            dw = np.sqrt(dt) * (z @ l_matrix.T)
            data = np.vstack([np.zeros((1, self.num_gens)), np.cumsum(dw, axis=0)])
            bm = pd.DataFrame(data, columns=np.arange(1, self.num_gens + 1, 1))
        else:
            norm_rv = np.random.normal(0, np.sqrt(dt), (self.n - 1, self.num_gens))  # N(0, dt)
            data = np.vstack((np.zeros((1, self.num_gens)), np.cumsum(norm_rv, axis=0)))
            bm = pd.DataFrame(data, columns=np.arange(1, self.num_gens + 1, 1))

        self.bm = bm
            
        return self.bm
    
    # ---------------------------------------------------------------------------------------------------
    
    def plot_bm(self):

        if not hasattr(self, "bm"):
            self.simulate()

        plt.figure(figsize=(10, 5), facecolor='darkgrey')
        plt.gca().set_facecolor("white")
        plt.plot(self.bm)
        plt.xlabel("t", fontsize=15)
        plt.ylabel("W(t)", fontsize=15)
        plt.grid()

        return plt.show()
    
    # ---------------------------------------------------------------------------------------------------

    def quadratic_variation(self):

        path = self.simulate()
        
        qv_bm = np.zeros((self.n-1, self.num_gens))
        self.qv_bm = qv_bm

        for j in range(self.num_gens):
            self.qv_bm[:, j] = np.cumsum((np.diff(path.iloc[:, j]))**2, axis = 0)
        self.qv_bm = pd.DataFrame(self.qv_bm, columns=np.arange(1, self.num_gens + 1, 1))

        return self.qv_bm
    
    # ---------------------------------------------------------------------------------------------------
    
    def plot_qv(self):

        if not hasattr(self, "qv_bm"):
            self.quadratic_variation()

        dt = self.t/(self.n-1); x = np.arange(0, self.t, dt)
        plt.figure(figsize=(10, 5), facecolor='darkgrey')
        plt.gca().set_facecolor("white")
        plt.plot(x, self.qv_bm)
        plt.xlabel('t', fontsize=15)
        plt.ylabel('Quadratic Variation', fontsize=15)
        plt.grid()

        return plt.show()
    
    # ---------------------------------------------------------------------------------------------------
