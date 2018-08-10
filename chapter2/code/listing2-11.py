from math import exp
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas import DataFrame
from pylab import *


datafile = os.path.join(os.getcwd(),
                        '../../datasets/abalone.data')

# read the data
abalone = pd.read_csv(datafile, header=None, prefix='V')
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'While weight',
                   'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']


# get summary for scaling
summary = abalone.describe()
min_rings = summary.iloc[3, 7]
max_rings = summary.iloc[7, 7]
nrows = len(abalone.index)

for i in range(nrows):
    # plot rows of data if they were series data
    data_row = abalone.iloc[i, 1:8]
    label_color = (abalone.iloc[i, 8] - min_rings) / (max_rings - min_rings)
    data_row.plot(color=plt.cm.RdYlBu(label_color), alpha=0.5)

plt.xlabel('Attribute Index')
plt.ylabel('Attribute Values')
plt.show()

# re-normalize using mean and S.D., then compress with logit function
mean_rings = summary.iloc[1, 7]
sd_rings = summary.iloc[2, 7]

for i in range(nrows):
    # plot data rows as series
    data_row = abalone.iloc[i, 1:8]
    norm_target = (abalone.iloc[i, 8] - mean_rings) / sd_rings
    label_color = 1.0 / (1.0 + exp(-norm_target))
    data_row.plot(color=plt.cm.RdYlBu(label_color), alpha=0.5)

plt.xlabel('Attribute index')
plt.ylabel('Attribute values')
plt.show()
