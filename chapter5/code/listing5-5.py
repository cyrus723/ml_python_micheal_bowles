import os
import numpy as np
from math import sqrt
from sklearn.linear_model import enet_path
import matplotlib.pyplot as plt


def getNtileBoundaries(data, ntiles):
    percentBdry = []
    for i in range(ntiles+1):
        percentBdry.append(np.percentile(data, i*(100)/ntiles))
    return percentBdry


x_list = []
labels = []
# Open the data set
datafile = os.path.join(os.getcwd(),
                        '../../datasets/sonar.all-data')
f = open(datafile, 'r')

for line in f.readlines():
    row = line.strip().split(',')
    x_list.append(row)

x_num = []
labels = []

for row in x_list:
    last_col = row.pop()
    if last_col == 'M':
        labels.append(1.0)
    else:
        labels.append(0.0)
    attr_row = [float(elt) for elt in row]
    x_num.append(attr_row)

# Number of rows and columns
n_row = len(x_num)
n_col = len(x_num[1])

alpha = 1.0

# calculate means and variances
x_means = []
x_sd = []
for i in range(n_col):
    col = [x_num[j][i] for j in range(n_row)]
    mean = sum(col)/n_row
    x_means.append(mean)
    col_diff = [(x_num[j][i] - mean) for j in range(n_row)]
    sum_sq = sum([col_diff[i] ** 2 for i in range(n_row)])
    stdev = sqrt(sum_sq/n_row)
    x_sd.append(stdev)

# Use calculate mean and standard deviation to normalize x_num
x_norm = []
for i in range(n_row):
    row_norm = [(x_num[i][j] - x_means[j])/x_sd[j] for j in range(n_col)]
    x_norm.append(row_norm)

# Normalize labels to center
mean_label = sum(labels)/n_row
sd_label = sqrt(sum([(labels[i] - mean_label) * (labels[i] - mean_label) \
                     for i in range(n_row)])/n_row)
labels_norm = [(labels[i] -mean_label)/sd_label]

# convert normalized labels to numpy array
Y = np.array(labels_norm)
# convert normalized attributes to numpy array
X = np.array(x_norm)

alphas, coefs, _ = enet_path(X, Y, l1_ratio=0.8, fit_intercept=False,
                            return_models=False)
plt.plot(alphas, coefs.T)
plt.xlabel('alpha')
plt.ylabel('coefficients')
plt.axis('tight')
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


# Make up names for columns in X
names = ['V' + str(i) for i in range(n_col)]
name_list = [names[nz_list[i]] for i in range(len(nz_list))]
print('Attributes ordered by how early they enter the model')

print(name_list)
print('')
# find coefficients corresponding to best alpha value. alpha value
# corresponding to normalized X and normalized Y is 0.020334883589342503
alpha_star = 0.020334883589342503
idx_lt_alpha_star = [i for i in range(100) if alphas[i] > alpha_star]
idx_star = max(idx_lt_alpha_star)

# here's the set of coefficients to deploy
coef_star = list(coefs[:, idx_star])
print('Best coefficient values ')
print(coef_star)

# The coefficients on normalized attributes give another slightly different
# ordering
abs_coef = [abs(a) for a in coef_star]
# sort by magnitude, in descending order
coef_sorted = sorted(abs_coef, reverse=True)
idx_coef_size = [abs_coef.index(a) for a in coef_sorted if a != 0.0]
names_list2 = [names[idx_coef_size[i]] for i in range(len(idx_coef_size))]

print('Attributes ordered by coef size at optimum alpha')
print(names_list2)
