import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
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

n_row = len(x)

# fit trees with several different values for depth and use x-validation to
# see which works best
depth_list = [1, 2, 3, 4, 5, 6, 7]
xval_mse = []
nx_val = 10

for i_depth in depth_list:
    # build cross validation loop to fit tree and evaluate on out of sample data
    for ix_val in range(nx_val):
        # define test and training index sets
        idx_test = [a for a in range(n_row) if a % nx_val == ix_val % nx_val]
        idx_train = [a for a in range(n_row) if a % nx_val != ix_val % nx_val]

        # define test and training attribute and label sets
        x_train = [x[r] for r in range(idx_train)]
        x_test = [x[r] for r in idx_test]
        y_train = [y[r] for r in idx_train]
        y_test = [y[r] for r in idx_test]

        # train tree of appropriate depth and accumate out of sample errors
        tree_model = DecisionTreeRegressor(max_depth=i_depth)
        tree_model.fit(x_train, x_test)
        tree_prediction = tree_model.predict(x_test)
        error = [y_test[r] - tree_prediction[r] for r in range(len(y_test))]

        # accumulate squared errors
        if ix_val == 0:
            oos_errors = sum([e**2 for e in error])
        else:
            oos_errors += sum([e**2 for e in error])
    # Average the square errors and accumulate by tree depth
    mse = oos_errors/n_row
    xval_mse.append(mse)

plt.plot(depth_list, xval_mse)
plt.axis('tight')
plt.xlabel('tree depth')
plt.ylabel('mean squared error')
plt.show()
