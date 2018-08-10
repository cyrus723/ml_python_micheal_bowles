import os
from math import log, cos
import numpy as np
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt

# read the data
dataset_file = os.path.join(os.getcwd(),
                            '../../datasets/winequality-red.csv')

x_list = []
labels = []
names = []
header = True

with open(dataset_file, 'r') as f:
    if header:
        names = f.readline().strip().split(';')
    for line in f.readlines():
        row = line.strip().split(';')
        labels.append(float(row[-1]))
        row.pop()
        float_row = [float(num) for num in row]
        x_list.append(float_row)

# Extend the alchohol variable, the last column in attr. matrix
x_extended = []
alch_col = len(x_list[1])
for row in x_list:
    # Adding new elements to the row
    new_row = list(row)
    alch = row[alch_col - 1]
    new_row.append(((alch-7) ** 2) / 10)
    new_row.append(5*log(alch-7))
    new_row.append(cos(alch))
    # add extended rows with new elements to the data
    x_extended.append(new_row)

n_row = len(x_list)
v1 = [x_extended[j][alch_col - 1] for j in range(n_row)]

for i in range(4):
    v2 = [x_extended[j][alch_col - 1 + i] for j in range(n_row)]
    plt.scatter(v1, v2)

plt.xlabel("Alcohol")
plt.ylabel("Extension functions of Alchoal")
plt.show()