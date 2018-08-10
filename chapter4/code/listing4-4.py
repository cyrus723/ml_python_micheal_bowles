import os
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

# Normalize columns in X and labels
nrows = len(x_list)
ncols = len(x_list[0])

# Calculate means and variance per attribute/column
x_means = []
x_sd = []
for i in range(ncols):
    col = [x_list[j][i] for j in range(nrows)]
    mean = sum(col)/nrows
    x_means.append(mean)
    col_diff = [(x_list[j][i] - mean) for j in range(nrows)]
    sum_sq = sum([col_diff[i] * col_diff[i] for i in range(nrows)])
    stdev = sqrt(sum_sq/nrows)
    x_sd.append(stdev)

# Use calculate mean and standard deviation to normalize x_list
x_norm = []
for i in range(nrows):
    row_norm = [(x_list[i][j] - x_means[j])/x_sd[j] for j in range(ncols)]
    x_norm.append(row_norm)

# Normalize labels
mean_label = sum(labels)/nrows
sd_label = sqrt(sum([(labels[i] - mean_label)**2 for i in range(nrows)])/nrows)

label_norm = [(labels[i] - mean_label)/sd_label for i in range(nrows)]

# Initialize matrix of betas at each step
beta = [0.0] * ncols
# initialize matrix of betas at each step
beta_mat = []
beta_mat.append(list(beta))

# number of steps to take
n_steps = 350
step_size = 0.004

for i in range(n_steps):
    # calculate residuals
    residuals = [0.0] * nrows       # difference between observed and predicted
    # Predictive method: multiplying with corresponding beta
    for j in range(nrows):
        labels_hat = sum([x_norm[j][k] * beta[k] for k in range(ncols)])
        residuals[j] = label_norm[j] - labels_hat

    # calculate correlation between attribute columns from normalized wine
    # and residuals
    corr = [0.0] * ncols
    for j in range(ncols):
        corr[j] = sum([x_norm[k][j] * residuals[k] for k in range(nrows)]) / \
                  nrows
    i_star = 0
    corr_star = corr[0]

    for j in range(1, (ncols)):
        if abs(corr_star) < abs(corr[j]):
            i_star = j
            corr_star = corr[j]

    beta[i_star] += (step_size * corr_star) / abs(corr_star)
    beta_mat.append(list(beta))

for i in range(ncols):
    # plot range of beta values for each attribute
    coef_curve = [beta_mat[k][i] for k in range(n_steps)]
    xaxis = range(n_steps)
    plt.plot(xaxis, coef_curve)

plt.xlabel('Steps taken')
plt.ylabel('coefficient values')
plt.show()
