import os
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import roc_curve, auc
from math import sqrt
import matplotlib.pyplot as plt

# Read the dataset
dataset_file = os.path.join(os.getcwd(),
                            '../../datasets/sonar.all-data')
x_list = []
labels = []
with open(dataset_file, 'r') as f:
    for line in f.readlines():
        # split comma separated row
        row = line.strip().split(',')
        # Assign labels
        if row[-1] == 'M':
            labels.append(1.0)
        else:
            labels.append(0.0)
        # Remove label from row
        row.pop()
        # convert row to floats
        float_row = [float(num) for num in row]
        x_list.append(float_row)

# divide attribute matrix and label vector into training (2/3) and testing
# (1/3) data
indices = range(len(x_list))
xlist_test = [x_list[i] for i in indices if i % 3 == 0]
xlist_train = [x_list[i] for i in indices if i % 3 != 0]
labels_test = [labels[i] for i in indices if i % 3 == 0]
labels_train = [labels[i] for i in indices if i % 3 != 0]

# Make numpy array (using list of lists) to match input class
x_train = np.array(xlist_train)
y_train = np.array(labels_train)

x_test = np.array(xlist_test)
y_test = np.array(labels_test)

alpha_list = [0.1**i for i in [-3, -2, -1, 0, 1, 2, 3, 4, 5]]
auc_list = []

for alph in alpha_list:
    rvm_ridge_model = linear_model.Ridge(alpha=alph)
    rvm_ridge_model.fit(x_train, y_train)
    fpr, tpr, threshold = roc_curve(y_test, rvm_ridge_model.predict(x_test))
    roc_auc = auc(fpr, tpr)
    auc_list.append(roc_auc)

print('AUC, Alpha')
for i in range(len(auc_list)):
    print(auc_list[i], alpha_list[i])

# Plot auch values versus alpha values
x = [-3, -2, -1, 0, 1, 2, 3, 4, 5]
plt.plot(x, auc_list)
plt.xlabel('-log(alpha)')
plt.ylabel('AUC')
plt.show()

# visualize the performance of the best classifier
index_best = auc_list.index(max(auc_list))
alph = alpha_list[index_best]
rvm_ridge_model = linear_model.Ridge(alpha=alph)
rvm_ridge_model.fit(x_train, y_train)

# scatter plot of actual vs predicted
plt.scatter(rvm_ridge_model.predict(x_test), y_test, s=100, alpha=0.25)
plt.xlabel('Predicted value')
plt.ylabel('Actual value')
plt.show()
