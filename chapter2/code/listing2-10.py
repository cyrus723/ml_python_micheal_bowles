import os
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

datafile = os.path.join(os.getcwd(),
                        '../../datasets/sonar.all-data')

rocks_vs_mines = pd.read_csv(datafile, header=None, prefix='V')

# Calculate corelations between real-valued attributes
correlation_mat = DataFrame(rocks_vs_mines.corr())

# visualize correlation using heatmap
plt.pcolor(correlation_mat)
plt.show()
