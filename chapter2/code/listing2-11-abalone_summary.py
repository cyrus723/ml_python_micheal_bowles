import matplotlib.pyplot as plt
import os
import pandas as pd
from pylab import *

datafile = os.path.join(os.getcwd(), '../../dataset/abalone/abalone.data')

# read the data
abalone = pd.read_csv(datafile, header=None, prefix='V')
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'While weight',
                   'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

# print head and tail
print(abalone.head())
print(abalone.tail())

# print summary
summary = abalone.describe()
print(summary)

# box plot for real-valued attributes
# convert to array for plot routine
# All rows, column 1:9, values return only the values, without column heading
array = abalone.iloc[:, 1:9].values
boxplot(array)
plt.xlabel('Attribute index')
plt.ylabel('quartile ranges')
show()

# Remove last column (it is out of scale) and replot
array2 = abalone.iloc[:, 1:8].values
boxplot(array2)
plt.xlabel('Attribute index')
plt.ylabel('Quartile ranges')
show()

# Re-normalize columns to zero mean and unit S.D. to have better generalization
# commonly used in k-means clustering, kNN etc.
abalone_normalized = abalone.iloc[:, 1:9]

for i in range(8):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]

    abalone_normalized.iloc[: i:(i+1)] = \
        (abalone_normalized.iloc[:, i:(i+1)] - mean) / sd

array3 = abalone_normalized.values
boxplot(array3)
plt.xlabel('Attribute index')
plt.ylabel('Quartile ranges - normalized')
show()
