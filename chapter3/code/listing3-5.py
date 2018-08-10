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
header = []

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

xtrain = np.array(xlist_train)
ytrain = np.array(labels_train)
xtest = np.array(xlist_test)
ytest = np.array(labels_test)

alpha_list = [0.1**i for i in list(range(7))]

rms_err = []
for alph in alpha_list:
    wine_ridge_model = linear_model.Ridge(alpha=alph)
    wine_ridge_model.fit(xtrain, ytrain)
    rms_err.append(np.linalg.norm((ytest-wine_ridge_model.predict(xtest)),
                                  2)/sqrt(len(ytest)))

print('RMS Error, alpha')
for i in range(len(rms_err)):
    print(rms_err[i], alpha_list[i])

# Plot curve of out-of-sample error versus alpha
x = range(len(rms_err))
plt.plot(x, rms_err, 'k')
plt.xlabel('-log(alpha)')
plt.ylabel('Error (RMS)')
plt.show()

# Plot histogram of out of sample errors for best alpha values and scatter
# lot of actual versus predicted

index_best = rms_err.index(min(rms_err))
alph = alpha_list[index_best]
wine_ridge_model = linear_model.Ridge(alpha=alph)
wine_ridge_model.fit(xtrain, ytrain)
err_vec = ytest -wine_ridge_model.predict(xtest)

plt.hist(err_vec)
plt.xlabel('Bin Boundaries')
plt.ylabel('Counts')
plt.show()

plt.scatter(wine_ridge_model.predict(xtest), ytest, s=100, alpha=0.10)
plt.xlabel('Predicted taste score')
plt.ylabel('Actual taste score')
plt.show()
