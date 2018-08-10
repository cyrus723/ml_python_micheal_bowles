import os
import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plt

datafile = os.path.join(os.getcwd(),
                        '../../datasets/glass.data')

# read the data
glass = pd.read_csv(datafile, header=None, prefix='V')
glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe',
                 'Type']
summary = glass.describe()
print(summary)

ncol = len(glass.columns)
glass_normalized = glass.iloc[:, 1:ncol]
ncol_norm = len(glass_normalized.columns)
summary_norm = glass_normalized.describe()

for i in range(ncol_norm):
    mean = summary_norm.iloc[1, i]
    sd = summary_norm.iloc[2, i]

    glass_normalized.iloc[:, i:(i+1)] = \
        (glass_normalized.iloc[:, i:(i+1)] - mean) / sd

array = glass_normalized.values
boxplot(array)
plt.xlabel("Attribute Index")
plt.ylabel("Quartile Ranges - Normalized")
plt.show()
