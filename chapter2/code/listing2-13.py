import os
import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plt

datafile = os.path.join(os.getcwd(),
                        '../../datasets/winequality-red.csv')

# read the data
wine = pd.read_csv(datafile, header=0, sep=';')
summary = wine.describe()
print(summary)

wine_normalized = wine
ncols = len(wine_normalized.columns)
for i in range(ncols):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]
    wine_normalized.iloc[:, i:(i+1)] = \
        (wine_normalized.iloc[:, i:(i+1)] - mean) / sd

array = wine_normalized.values
boxplot(array)
plt.xlabel('Attribute Index')
plt.ylabel('Quartile Ranges- Normalized')
plt.show()
