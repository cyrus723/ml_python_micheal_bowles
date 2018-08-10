import os
from math import sqrt
import numpy as np
from sklearn.linear_model import enet_path
import matplotlib.pyplot as plt


# read the data
datafile = os.path.join(os.getcwd(),
                        '../../datasets/sonar.all-data')
f = open(datafile, 'r')

# arrrange data into list for labels and list of lists for attributes
x_list = []
for line in f.readlines():
    row = line.strip().split(',')
    x_list.append(row)

names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']

# separate attributes and labels
x_num = []
labels = []
for row in x_list:
    labels.append(row.pop())        # Extract last element at index= -1
    l = len(row)
    attr_row = [float(row[i]) for i in range(1, l)]
    x_num.append(attr_row)

# number of rows and columns in X matrix
n_row = len(x_num)
n_col = len(x_num[1])

# Create one vs all label vectors
# get distinct glass types and assign index to each
y_one_all = []
label_set = set(labels)
label_list = list(label_set)
label_list.sort()
n_labels = len(label_list)
for i in range(n_row):
    y_row = [0.0] * n_labels
    index = label_list.index(labels[i])
    y_row[index] = 1.0
    y_one_all.append(y_row)

# calculate means and vairances
x_means = []
x_sd = []
for i in range(n_col):
    col = [x_num[j][i] for j in range(n_row)]
    mean = sum(col) / n_row
    x_means.append(mean)
    col_diff = [(x_num[j][i] - mean) for j in range(n_row)]
    sum_sq = sum([col_diff[i] * col_diff[i] for i in range(n_row)])
    stdev = sqrt(sum_sq / n_row)
    x_sd.append(stdev)
# use calculate mean and standard deviation to normalize x_num
x_norm = []
for i in range(n_row):
    row_norm = [(x_num[i][j] - x_means[j]) / x_sd[j]
                for j in range(n_col)]
x_norm.append(row_norm)
# normalize y's to centeror
y_means = []
y_sd = []
for i in range(n_labels):
    col = [y_one_all[j][i] for j in range(n_row)]
    mean = sum(col) / n_row
    y_means.append(mean)
    col_diff = [(y_one_all[j][i] - mean) for j in range(n_row)]
    sum_sq = sum([col_diff[i] * col_diff[i] for i in range(n_row)])
    stdev = sqrt(sum_sq / n_row)
    y_sd.append(stdev)

y_norm = []
for i in range(n_row):
    row_norm = [(y_one_all[i][j] - y_means[j])/y_sd[j] for j in range(n_labels)]
    y_norm.append(row_norm)

# number of cross validation folds
nx_val = 10
n_alphas = 200
mis_class = [0.0] * n_alphas

for ix_val in range(nx_val):
    # define test and training index sets
    idx_test = [a for a in range(n_row) if a % nx_val == ix_val % nx_val]
    idx_train = [a for a in range(n_row) if a % nx_val != ix_val % nx_val]

    # define test and training attribute and label sets
    x_train = np.array([x_norm[r] for r in idx_train])
    x_test = np.array([x_norm[r] for r in idx_test])

    y_train = [y_norm[r] for r in idx_train]
    y_test = [y_norm[r] for r in idx_test]
    labels_test = [labels[r] for r in idx_test]

    # build model for each column in y_train
    models = []
    len_train = len(y_train)
    len_test = n_row - len_train
    for i_model in range(n_labels):
        y_temp = np.array([y_train[j][i_model] for j in range(len_train)])
        models.append(enet_path(x_train, y_temp, l1_ratio=1.0,
                                fit_intercept=False, eps=0.5e-3,
                                n_alphas=n_alphas, return_models=False))
    for i_step in range(1, n_alphas):
        # Assemble the predictions for all the models, find largest
        # predictions and calc error
        all_predictions = []
        for i_model in range(n_labels):
            _, coefs, _ = models[i_model]
            pred_temp = list(np.dot(x_test, coefs[:, i_step]))
            # Un-normalize the prediction for comparison
            pred_un_norm = [(pred_temp[j] * y_sd[i_model] + y_means[i_model])
                            for j in range(len(pred_temp))]
            all_predictions.append(pred_un_norm)
        predictions = []
        for i in range(len_test):
            list_of_pred = [all_predictions[j][i] for j in range(n_labels)]
            idx_max = list_of_pred.index(max(list_of_pred))
            if label_list[idx_max] != labels_test:
                mis_class[i_step] += 1.0

mis_class_plot = [mis_class[i]/n_row for i in range(1, n_alphas)]
plt.plot(mis_class_plot)
plt.xlabel('Penalty parameter steps')
plt.ylabel('Misclassification error rate')
plt.show()
