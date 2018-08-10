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

# number of samples to draw for stochastic bagging
n_bag_samples = int(len(x_train) * 0.5)

for i_trees in range(num_tree_max):
    idx_bag = []
    for i in range(n_bag_samples):
        idx_bag.append(random.choice(range(len(x_train))))
    x_train_bag = [x_train[i] for i in idx_bag]
    y_train_bag = [y_train[i] for i in idx_bag]

    model_list.append(DecisionTreeRegressor(max_depth=tree_depth))
    model_list[-1].fit(x_train_bag, y_train_bag)

    # make prediction with latest model and add to list of predictions
    latest_prediction = model_list[-1].predict(x_test)
    pred_list.append(list(latest_prediction))

# Build cumulative prediction from the first "n" models
mse = []
all_predictions = []
for i_models in range(len(model_list)):
    # average first "i_models" of the predictions
    prediction = []
    for i_pred in range(len(x_test)):
        prediction.append(sum([pred_list[i][i_pred]
                               for i in range(i_models + 1)]) / (i_models + 1))
    all_predictions.append(prediction)
    errors = [(y_test[i] - prediction[i]) for i in range(len(y_test))]
    mse.append(sum([e**2 for e in errors])/len(y_test))

n_models = [i+1 for i in range(len(model_list))]
plt.plot(n_models, mse)
plt.axis('tight')
plt.xlabel('Number of tree models in ensemble')
plt.ylabel('Mean squared error')
plt.ylim((0.0, max(mse)))
plt.show()

print('Minimum MSE')
print(min(mse))
