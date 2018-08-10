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

glass_normalized = glass
ncols = len(glass_normalized.columns)
nrows = len(glass_normalized.index)
summary = glass_normalized.describe()
n_data_col = ncols-1

print(summary)
# Normalize except for labels
for i in range(ncols - 1):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]
    glass_normalized.iloc[:, i:(i+1)] = \
        (glass_normalized.iloc[:, i:(i+1)] - mean) / sd

for i in range(nrows):
    data_row = glass_normalized.iloc[i, 1:n_data_col]
    label_color = glass_normalized.iloc[i, 1:n_data_col]/7.0
    data_row.plot(color=plt.cm.RdYlBu(label_color), alpha=0.5)

plt.xlabel('Attribute Index')
plt.ylabel('Attribute values')
plt.show()
