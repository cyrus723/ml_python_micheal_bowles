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

plt.plot(x_plot, y)
plt.axis('tight')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

simple_tree = DecisionTreeRegressor(max_depth=1)
simple_tree.fit(x, y)

# draw the tree
with open('simple_tree.dot', 'w') as f:
    f = tree.export_graphviz(simple_tree, out_file=f)

# Compare prediction from tree with true values
y_hat = simple_tree.predict(x)

plt.figure()
plt.plot(x_plot, y, label='True y')
plt.plot(x_plot, y_hat, label='Tree prediction', linestyle='--')
plt.legend(bbox_to_anchor=(1, 0.2))
plt.axis('tight')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

simple_tree2 = DecisionTreeRegressor(max_depth=2)
simple_tree2.fit(x, y)

# draw the tree
y_hat = simple_tree2.predict(x)

plt.figure()
plt.plot(x_plot, y, label='True y')
plt.plot(x_plot, y_hat, label='Tree Prediction', linestyle='--')
plt.legend('tight')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# split point calculations-try every possible split point to find best one
sse = []
x_min = []
for i in range(1, len(x_plot)):
    # divide list into points on left and right of the split point
    lh_list = list(x_plot[0: i])
    rh_list = list(x_plot[i: len(x_plot)])

    # Calculate averages on each side
    lh_avg = sum(lh_list) / len(lh_list)
    rh_avg = sum(lh_list) / len(lh_list)

    # calculate sum square error on left and right and total
    lh_sse = sum([(s - lh_avg) ** 2 for s in lh_list])
    rh_sse = sum([(s - rh_avg) ** 2 for s in rh_list])

    # add sum of left and right to list of errors
    sse.append(lh_sse + rh_sse)
    x_min.append(max(lh_list))

plt.plot(range(1, len(x_plot)), sse)
plt.xlabel('Split point index')
plt.ylabel('Sum squared error')
plt.show()

min_sse = min(sse)
idx_min = sse.index(min_sse)
print(x_min[idx_min])

# what happens if the depth is really high?
simple_tree6 = DecisionTreeRegressor(max_depth=6)
simple_tree6.fit(x, y)

# too many nodes to draw the tree
# with open('simple_tree6.dot', 'w') as f:
#     f = tree.export_graphviz(simple_tree6, out_file=f

# compare prediction from tree with true values
y_hat = simple_tree6.predict(x)

plt.figure()
plt.plot(x_plot, y, label='True y')
plt.plot(x_plot, y_hat, label='Tree prediction', linestyle='--')
plt.legend(bbox_to_anchor=(1, 0.2))
plt.axis('tight')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
