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

# Normalize columns in x and labels
n_rows = len(x_list)
n_cols = len(x_list[0])

# calculate means and variances
x_means = []
x_sd = []
for i in range(n_cols):
    col = [x_list[j][i] for j in range(n_rows)]
    mean = sum(col)/n_rows
    x_means.append(mean)
    col_diff = [(x_list[j][i] - mean) for j in range(n_rows)]
    sum_sq = sum([col_diff[i] ** 2 for i in range(n_rows)])
    stdev = sqrt(sum_sq/n_rows)
    x_sd.append(stdev)

# Use calculated mean and standard deviation to normalize x_list
x_norm = []
for i in range(n_rows):
    row_norm = [(x_list[i][j] -x_means[j])/x_sd[j] for j in range(n_cols)]
    x_norm.append(row_norm)

# Normalize labels
mean_label = sum(labels)/n_rows
sd_label = sqrt(sum([(labels[i] - mean_label) ** 2 for i in range(
    n_rows)])/n_rows)
labels_norm = [(labels[i] - mean_label)/sd_label for i in range(n_rows)]

# convert list of list to np array for input to sklearn packages
# unnormalize labels
Y = np.array(labels)
# normalized labels
Y = np.array(labels_norm)

# unnormalied X
X = np.array(x_list)
# normalized x
X = np.array(x_norm)

# call lasso CV from sk_learn.linear_model
alphas, coefs, _ = linear_model.lasso_path(X, Y, return_models=False)

plt.plot(alphas, coefs.T)

plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.axis('Tight')
plt.semilogx()

ax = plt.gca()
ax.invert_xaxis()
plt.show()

n_attr, n_alpha = coefs.shape
# Find coefficient ordering
nz_list = []
for i_alpha in range(1, n_alpha):
    coef_list = list(coefs[:, i_alpha])
    nz_coef = [i for i in range(n_attr) if coef_list[i] != 0.0]
    for q in nz_coef:
        if q not in nz_list:
            nz_list.append(q)

name_list = [names[nz_list[i]] for i in range(len(nz_list))]
print('Attributes ordered by how early they enter the model', name_list)

# find coefficients corresponding to the best alpha value
alpha_star = 0.013561387700964642   # hard coded to get best results

idx_lt_alpha_star = [i for i in range(100) if alphas[i] < alpha_star]
idx_star = max(idx_lt_alpha_star)

# heres the set of coefficients to deploy
coef_star = list(coefs[:, idx_star])
print('Best coefficients values: ', coef_star)

# The coefficients on normalized attrbutes give another slight different
# ordering
abs_coef = [abs(a) for a in coef_star]

# sort by magnitude: Descending order
coef_sorted = sorted(abs_coef, reverse=True)

idx_coef_size = [abs_coef.index(a) for a in coef_sorted if a != 0.0]

names_list2 = [names[idx_coef_size[i]] for i in range(len(idx_coef_size))]

print('Attributes order by coef size at optimum alpha: ', names_list2)
