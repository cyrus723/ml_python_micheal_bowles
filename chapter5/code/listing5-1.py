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
wine_model = linear_model.LassoCV(cv=10).fit(X, Y)

# Display results
plt.figure()
plt.plot(wine_model.alphas_, wine_model.mse_path_, ':')
plt.plot(wine_model.alphas_, wine_model.mse_path_.mean(axis=-1),
         label='Average MSE Across Folds', linewidth=2)
plt.axvline(wine_model.alpha_, linestyle='--',
            label='CV estimate of best alpha')
plt.semilogx()
plt.legend()
ax = plt.gca()
ax.invert_xaxis()
plt.xlabel('Alpha')
plt.ylabel('Mean Square Error')
plt.axis('tight')
plt.show()

# Print out the value of alpha that minimizes the CV error
print('Alpha value that minimizes CV error ', wine_model.alpha_)
print('Minimum MSE ', min(wine_model.mse_path_.mean(axis=-1)))
