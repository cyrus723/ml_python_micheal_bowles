import matplotlib.pyplot as plt
import os
import random
from sklearn.tree import DecisionTreeRegressor


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

# Take fixed set 30% of sample
random.seed(1)
n_sample = int(n_rows * 0.30)
idx_test = random.sample(range(n_rows), n_sample)
idx_test.sort()
idx_train = [i for i in range(n_rows) if i not in idx_test]

# Define test and training attribute and label sets
x_train = [x_list[r] for r in idx_train]
x_test = [x_list[r] for r in idx_test]
y_train = [labels[r] for r in idx_train]
y_test = [labels[r] for r in idx_test]

# Train a series of models on random subsets of the training data
# collect the models in a list and check error of composite as list grows

# maximum number of models to generaete
num_tree_max = 30

# Tree depth - typically at the high end
tree_depth = 1

# initialize a list to hold models
model_list = []
pred_list = []
eps = 0.1

# Initialize residuals to be labels y
residuals = list(y_train)

for i_trees in range(num_tree_max):
    model_list.append(DecisionTreeRegressor(max_depth=tree_depth))
    model_list[-1].fit(x_train, residuals)

    # Make prediction wiht latest model and add to the list of predictions
    latest_in_sample_pred = model_list[-1].predict(x_train)

    # Use new predictions to update residuals
    residuals = [residuals[i] - eps * latest_in_sample_pred[i]
                 for i in range(len(residuals))]

    latest_out_of_sample_pred = model_list[-1].predict(x_test)
    pred_list.append(list(latest_out_of_sample_pred))

# Build cumulative predictions from first "n" models
mse = []
all_pred = []
for i_models in range(len(model_list)):
    # add the first "i_models" of the predictions and multiply by eps
    pred = []
    for i_pred in range(x_test):
        pred.append(sum([pred_list[i][i_pred] for i in range(i_models+1)]) *
                    eps)
    all_pred.append(pred)

    errors = [(y_test[i] - pred[i]) for i in range(len(y_test))]
    mse.append(sum([e**2 for e in errors])/len(y_test))

n_models = [i+1 for i in range(model_list)]

plt.plot(n_models, mse)
plt.axis('tight')
plt.xlabel('Number of trees in ensemble')
plt.ylabel('Mean squared error')
plt.ylim((0.0, max(mse)))
plt.show()

print('Minimum MSE: ', min(mse))
