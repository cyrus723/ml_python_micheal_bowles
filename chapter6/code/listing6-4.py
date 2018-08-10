import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.tree import DecisionTreeRegressor

# Build a simple dataset  with y=x+random
n_points = 100

# X values for plotting
x_plot = [(float(i) / float(n_points) - 0.5) for i in range(n_points + 1)]

# x needs to be list of lists
x = [[s] for s in x_plot]

# y labels has random noise added to x_values
# Set seed
np.random.seed(1)
y = [s + np.random.normal(scale=0.1) for s in x_plot]

# Take fixed test set 30% of sample
n_sample = int(n_points * 0.30)
idx_test = random.sample(range(n_points), n_sample)
idx_test.sort()
idx_train = [i for i in range(n_points) if i not in idx_test]

# define test and training attribute and label sets
x_train = [x[r] for r in idx_train]
x_test = [x[r] for r in idx_test]
y_train = [y[r] for r in idx_train]
y_test = [y[r] for r in idx_test]

# Train a series of models on random subsets of the training data
# collect the models in a list and check error of composite as list grows

# maximum number of models to generate
num_tree_max = 20

# tree depth - typically at the high end
tree_depth = 1

# initialize a list to hold models
model_list = []
pred_list = []

# number of samples to draw for stochastic bagging
n_bag_samples = int(len(x_train) * 0.5)

for i_trees in range(num_tree_max):
    idx_bag = random.sample(range(len(x_train)), n_bag_samples)
    x_train_bag = [x_train[i] for i in idx_bag]
    y_train_bag = [y_train[i] for i in idx_bag]

    model_list.append(DecisionTreeRegressor(max_depth=tree_depth))
    model_list[-1].fit(x_train_bag, y_train_bag)

    # make prediction with latest model and add to list of predictions
    latest_prediction = model_list[-1].predict(x_test)
    pred_list.append(list(latest_prediction))

# build cumulative prediction from first n models
mse = []
all_predictions = []
for i_models in range(len(model_list)):
    # average first "i_models" of the predictions
    prediction = []
    for i_pred in range(len(x_test)):
        prediction.append(sum([pred_list[i][i_pred] \
                               for i in range(i_models + 1)]) / (i_models + 1))
    all_predictions.append(prediction)
    errors = [(y_test[i] - prediction[i]) for i in range(len(y_test))]
    mse.append(sum([e * e for e in errors]) / len(y_test))

n_models = [i + 1 for i in range(len(model_list))]
plt.plot(n_models, mse)
plt.axis('tight')
plt.xlabel('Number of models in ensemble')
plt.ylabel('mean squared error')
plt.ylim((0.0, max(mse)))
plt.show()

plot_list = [0, 9, 19]
for i_plot in plot_list:
    plt.plot(x_test, all_predictions[i_plot])
plt.plot(x_test, y_test, linestyle='--')
plt.axis('tight')
plt.xlabel('x values')
plt.ylabel('predictions')
plt.show()
