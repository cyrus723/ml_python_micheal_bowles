import os
import numpy as np
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt


def xattr_select(x, idx_set):
    # Takes X matrix and returns subset containing columns in idx_set
    x_out = []
    for row in x:
        x_out.append([row[i] for i in idx_set])
    return x_out


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

# Divide attributes and labels into training and test sets
indices = range(len(x_list))
xlist_test = [x_list[i] for i in indices if i % 3 == 0]
xlist_train = [x_list[i] for i in indices if i % 3 != 0]
labels_test = [labels[i] for i in indices if i % 3 == 0]
labels_train = [labels[i] for i in indices if i % 3 != 0]

# build list of attributes on at a time
attr_list = []
index = range(len(x_list[1]))
index_set = set(index)
index_seq = []
oos_err = []

for i in index:
    attr_set = set(attr_list)
    attr_try_set = index_set - attr_set
    attr_try = [ii for ii in attr_try_set]
    err_list = []
    attr_tmp = []
    # Try each choice to find oos
    for i_try in attr_try:
        attr_tmp = [] + attr_list
        attr_tmp.append(i_try)
        # use attr_temp to form training and testing sub matrices
        xtrain_tmp = xattr_select(xlist_train, attr_tmp)
        xtest_tmp = xattr_select(xlist_test, attr_tmp)
        # form into numpy arrays
        xtrain = np.array(xtrain_tmp)
        xtest = np.array(xtest_tmp)
        ytrain = np.array(labels_train)
        ytest = np.array(labels_test)

        # Use scikit learn linear regression
        wine_qmodel = linear_model.LinearRegression()
        wine_qmodel.fit(xtrain, ytrain)
        # Use trained model to generate prediction and calculate rmsError
        rms_err = np.linalg.norm((ytest-wine_qmodel.predict(xtest)), 2)/sqrt(
            len(ytest))
        err_list.append(rms_err)
        attr_tmp = []
    i_best = np.argmin(err_list)
    attr_list.append(attr_try[i_best])
    oos_err.append(err_list[i_best])

print('Out of sample error vs. attribute set size: ', oos_err)
print('Best attribute indices: ', attr_list)
names_list = [names[i] for i in attr_list]
print('Best attribute names: ', names_list)

# Plot error versus number of attributes
x = range(len(oos_err))
plt.plot(x, oos_err, 'k')
plt.xlabel('Number of attributes')
plt.ylabel('Error (RMS)')
plt.show()

# Plot histogram of OOS errors
# Identify index corresponding to min value,
# retrain with the corresponding attributes
# Use resulting model to predict against out of sample data.
# Plot errors (aka residuals)
index_best = oos_err.index(min(oos_err))
attr_best = attr_list[1:(index_best+1)]

# Define column-wise subsets of xListTrain and xListTests and convert to numpy
xtrain_tmp = xattr_select(xlist_train, attr_best)
xtest_tmp = xattr_select(xlist_test, attr_best)
xtrain = np.array(xtrain_tmp)
xtest = np.array(xtest_tmp)

# Train and plot error histogram
wine_qmodel = linear_model.LinearRegression()
wine_qmodel.fit(xtrain, ytrain)
err_vec = ytest-wine_qmodel.predict(xtest)
plt.hist(err_vec)
plt.xlabel('Bin boundaries')
plt.ylabel('counts')
plt.show()

# Scatter plot of actual versus predicted
plt.scatter(wine_qmodel.predict(xtest), ytest, s=100, alpha=0.10)
plt.xlabel('Predicted taste score')
plt.ylabel('Actual taste score')
plt.show()
