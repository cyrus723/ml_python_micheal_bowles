import numpy as np
import os
import pylab
import scipy.stats as stats


attributes = []
labels = []
# Open the data set
datafile = os.path.join(os.getcwd(),
                        '../../datasets/sonar.all-data')
f = open(datafile, 'r')

for line in f.readlines():
    row = line.strip().split(',')
    attributes.append(row)


nrows= len(attributes)
ncol = len(attributes[0])

types = [0] * 3
col_counts = []

# generate summary statistics for column 3
col = 3
col_data = []
for row in attributes:
    col_data.append(float(row[col]))

stats.probplot(col_data, dist="norm", plot=pylab)
pylab.show()
