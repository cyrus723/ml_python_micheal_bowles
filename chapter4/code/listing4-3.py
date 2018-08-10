import os
import numpy as np
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt


def S(z, gamma):
    if gamma >= abs(z):
        return 0.0
    return (z/abs(z))*(abs(z)-gamma)


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

# Select value for alpha parameter
alpha = 1.0

# Make a pass through the data to determine value of lambda that just
# suppresses all coefficients
# Starting value for all betas is zero

xy = [0.0] * ncols
for i in range(nrows):
    for j in range(ncols):
        xy[j] += x_norm[i][j] * label_norm[i]

max_xy = 0.0
for i in range(ncols):
    val = abs(xy[i])/nrows
    if val > max_xy:
        max_xy = val

# Calculate starting value for lambda
lam = max_xy/alpha

# this value of lambda corresponds to beta = list of 0's initialize a vector
# of coefficients beta
beta = [0.0] * ncols

# Initialize matrix of betas at each step
beta_mat = []
beta_mat.append(list(beta))

# begin iteration
n_steps = 100
lam_mult = 0.93
nz_list = []
for i_step in range(n_steps):
    # Make lambda smaller so that some coefficient become non-zero
    lam = lam * lam_mult
    delta_beta = 100.0
    eps = 0.01
    iter_step = 0
    beta_inner = list(beta)
    while delta_beta > eps:
        iter_step += 1
        if iter_step > 100:
            break
        # cycle through attributes and update one at a time
        # record starting value for comparison
        beta_start = list(beta_inner)
        for i_col in range(ncols):
            xyj = 0.0
            for i in range(nrows):
                # calculate residual with current value of beta
                label_hat = sum([x_norm[i][k] * beta_inner[k]
                                 for k in range(ncols)])
                residual = label_norm[i] - label_hat
                xyj += x_norm[i][i_col] * residual
            unc_beta = xyj/nrows + beta_inner[i_col]
            beta_inner[i_col] = S(unc_beta, lam*alpha)/(1+lam*(1-alpha))
        sum_diff = sum([abs(beta_inner[n] - beta_start[n])
                        for n in range(ncols)])
        sum_beta = sum([abs(beta_inner[n]) for n in range(ncols)])
        delta_beta = sum_diff/sum_beta
    print(i_step, iter_step)
    beta = beta_inner

    # add newly determined beta to list
    beta_mat.append(beta)

    # keep track of the order in which the betas become non-zero
    nz_beta = [idx for idx in range(ncols) if beta[idx] != 0.0]
    for q in nz_beta:
        if not q in nz_list:
            nz_list.append(q)

# print out order list of betas
name_list = [names[nz_list[i]] for i in range(len(nz_list))]
print(name_list)

n_pts = len(beta_mat)
for i in range(ncols):
    # plot range of beta values for each attribute
    coef_curve = [beta_mat[k][i] for k in range(n_pts)]
    x_axis = range(n_pts)
    plt.plot(x_axis, coef_curve)

plt.xlabel('Steps taken')
plt.ylabel('Coefficient values')
plt.show()