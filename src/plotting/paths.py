import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def path_generator_plot(out, column_name: str = "W(t)", row_name: str = "t"):
    plt.figure(figsize=(10, 5), facecolor='darkgrey')
    plt.gca().set_facecolor("white")
    plt.plot(out)
    plt.xlabel(row_name, fontsize=15)
    plt.ylabel(column_name, fontsize=15)
    plt.grid()

    return plt.show()