import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import random

# build a simple data set with y=x+random
n_points = 1000

# x values for plotting
x_plot = [(float(i)/float(n_points) - 0.5) for i in range(n_points + 1)]

# X needs to be list of lsits
x = [[s] for s in x_plot]

# y (labels) has a random noise added to x-value
# set seed
np.random.seed(1)
y = [s+np.random.normal(scale=0.1) for s in x_plot]

# take fiixed test set 30% of sample
n_sample = int(n_points * .30)
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
num_trees_max = 30

# tree depth - typically at the high end
tree_depth = 5

# initialize a list of hold models
model_list = []
pred_list = []
eps = 0.3

# Initialize residuals to be the labels y
residuals = list(y_train)

for i_trees  in range(num_trees_max):
    model_list.append(DecisionTreeRegressor(max_depth=tree_depth))
    model_list[-1].fit(x_train, residuals)

    # make predictions with latest model and add to list of predictions
    latest_in_sample_pred = model_list[-1].predict(x_train)

    # use new predictions to update residuals
    residuals = [residuals[i] - eps * latest_in_sample_pred[i]
                 for i in range(len(residuals))]

    latest_out_sample_pred = model_list[-1].predict(x_test)
    pred_list.append(list(latest_out_sample_pred))

# build cumulative prediction from first "n" models
mse = []
all_predictions = []
for i_models in range(len(model_list)):
    # add the first "i_models' of the predictions and multiply by eps
    prediction = []
    for i_pred in range(len(x_test)):
        prediction.append(sum([pred_list[i][i_pred]
                               for i in range(i_models+1)]) * eps)
    all_predictions.append(prediction)
    errors = [(y_test[i] - prediction[i]) for i in range(len(y_test))]
    mse.append(sum([e**2 for e in errors])/len(y_test))

n_models = [i+1 for i in range(len(model_list))]


plt.plot(n_models, mse)
plt.axis('tight')
plt.xlabel('Number of models in ensemble')
plt.ylabel('mean squared error')
plt.ylim((0.0, max(mse)))
plt.show()

plot_list = [0, 14, 29]
line_type = [':', '-.', '--']
plt.figure()
for i in range(len(plot_list)):
    i_plot = plot_list[i]
    text_legend = 'Prediction with ' + str(i_plot) + ' Trees'
    plt.plot(x_test, all_predictions[i_plot], label=text_legend,
             linestyle=line_type[i])
plt.plot(x_test, y_test, label='True y value', alpha=0.25)
plt.legend(bbox_to_anchor=(1, 0.3))
plt.axis('tight')
plt.xlabel('x value')
plt.ylabel('predictions')
plt.show()
