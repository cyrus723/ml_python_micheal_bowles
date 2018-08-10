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

# Build cross validation loop to determine best coefficient values
# number of cross-validation folds

nx_val = 10

# number of steps
n_steps = 350
step_size = 0.004

# initialize list for storing errors
errors = []
for i in range(n_steps):
    b = []
    errors.append(b)

for ix_val in range(nx_val):
    # define test and training index set
    # The sample given in the book is wrong
    idx_test = [a for a in range(nrows) if a % nx_val == 0]
    idx_train = [a for a in range(nrows) if a % nx_val != 0]

# Define test and training attribute and label sets
x_train = [x_norm[r] for r in idx_train]
x_test = [x_norm[r] for r in idx_test]
label_train = [label_norm[r] for r in idx_train]
label_test = [label_norm[r] for r in idx_test]

# Train LARS regression on training data
nrows_train = len(idx_train)
nrows_test = len(idx_test)

# Initialize a vector of coefficients beta
beta = [0.0] * ncols

# Initialize matrix of betas at each step
beta_mat = []
beta_mat.append(list(beta))

for i_step in range(n_steps):
    # calculate residuals
    residuals = [0.0] * nrows
    # Predicting method
    for j in range(nrows_train):
        labels_hat = sum([x_train[j][k] * beta[k] for k in range(ncols)])
        residuals[j] = label_train[j] - labels_hat
    # calculate correlation between attribute columns for normalized wine and
    #  residual
    corr = [0.0] * ncols
    for j in range(ncols):
        corr[j] = sum([x_train[k][j] * residuals[k] for k in range(
            nrows_train)]) / nrows_train
    i_star = 0
    corr_star = corr[0]

    for j in range(1, (ncols)):
        if abs(corr_star) < abs(corr[j]):
            i_star = j
            corr_star = corr[j]

    beta[i_star] += step_size * corr_star / abs(corr_star)
    beta_mat.append(list(beta))

    # Use beta just calculate to predict and accumulate OOS error
    for j in range(nrows_test):
        labels_hat = sum([x_test[j][k] * beta[k] for k in range(ncols)])
        err = label_test[j] - labels_hat
        errors[i_step].append(err)

cv_curve = []
print(errors)
for err_vec in errors:
    mse = sum([x*x for x in err_vec])/len(err_vec)
    cv_curve.append(mse)

min_mse = min(cv_curve)
min_pt = [i for i in range(len(cv_curve)) if cv_curve[i] == min_mse][0]
print('Minimum mean square error: ', min_mse)
print('Index of minimum mean square error: ', min_pt)

xaxis = range(len(cv_curve))
plt.plot(xaxis, cv_curve)

plt.xlabel('Steps taken')
plt.ylabel('Mean square error')
plt.show()

