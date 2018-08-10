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

# take fixed test set 30% of sample
random.seed(1)

n_sample = int(n_rows * 0.30)
idx_test = random.sample(range(n_rows), n_sample)
idx_test.sort()
idx_train = [i for i in range(n_rows) if i not in idx_test]

# define test and training attributes and label sets
x_train = [x_list[r] for r in idx_train]
y_train = [labels[r] for r in idx_train]
x_test = [x_list[r] for r in idx_test]
y_test = [labels[r] for r in idx_test]


# train a series of models on random subsets of the training data
# collect the models in a list and check error of composite as list grows

# maximum number of models to generate
num_trees_max = 30

# tree depth - typically at the high end
tree_depth = 12

# pick how many attributes will be used in each model
# authors recommend 1/3 for regression problem
n_attr = 4

# initialize a list to hold models
model_list = []
index_list = []
pred_list = []
n_train_rows = len(y_train)

for i_trees in range(num_trees_max):
    model_list.append(DecisionTreeRegressor(max_depth=tree_depth))
    # take random sample of attributes
    idx_attr = random.sample(range(n_cols), n_attr)
    idx_attr.sort()
    index_list.append(idx_attr)

    # take a random sample of training rows
    idx_rows = []
    for i in range(int(0.5 * n_train_rows)):
        idx_rows.append(random.choice(range(len(x_train))))
    idx_rows.sort()

    # build training set
    x_rf_train = []
    y_rf_train = []

    for i in range(len(idx_rows)):
        temp = [x_train[idx_rows[i]][j] for j in idx_attr]
        x_rf_train.append(temp)
        y_rf_train.append(y_train[idx_rows[i]])

    model_list[-1].fit(x_rf_train, y_rf_train)

    # restrict x_test to attributes selected for training
    x_rf_test = []
    for xx in x_test:
        temp = [xx[i] for i in idx_attr]
        x_rf_test.append(temp)

    latest_out_sample_pred = model_list[-1].predict(x_rf_test)

# Build cumulative prediction from first "n" models
mse = []
all_pred = []
for i_models in range(len(model_list)):
    # add first "i_models" of predictions and multiply by eps
    pred = []
    for i_pred in range(len(x_test)):
        pred.append(sum([pred_list[i][i_pred]
                         for i in range(i_models+1)])/(i_models+1))
    all_pred.append(pred)
    errors = [(y_test[i] - pred[i]) for i in range(len(y_test))]
    mse.append(sum([e*e for e in errors])/len(y_test))

n_models = [i+1 for i in range(len(model_list))]

plt.plot(n_models, mse)
plt.axis('tight')
plt.xlabel('Number of trees in ensemble')
plt.ylabel('Mean squared error')
plt.ylim((0.0, max(mse)))
plt.show()

print('Minimum MSE: ', min(mse))
