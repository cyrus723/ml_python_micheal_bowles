import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error


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
x = np.array(x_list)
y = np.array(labels)
wine_names = np.array(names)

# take fixed holdout set 30% of data rows
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,
                                                    random_state=531)

# train Random forest at a range of ensemble sizes in order to see how the
# mse changes
mse_oos = []
n_tree_list = range(50, 500, 10)
for i_trees in n_tree_list:
    depth = None
    max_feat = 4        # can be tweaked
    wine_rf_model = ensemble.RandomForestRegressor(
        n_estimators=i_trees, max_depth=depth, max_features=max_feat,
        oob_score=False, random_state=531)
    wine_rf_model.fit(x_train, y_train)

    # accumulate mse on test set
    pred = wine_rf_model.predict(x_test)
    mse_oos.append(mean_squared_error(y_test, pred))

print('MSE: ', mse_oos[-1])

# Plot training and test errors vs number of trees in ensemble
plt.plot(n_tree_list, mse_oos)
plt.xlabel('Number of trees in ensemble')
plt.ylabel('Mean squared error')
plt.show()

# plot feature importance
feature_importance = wine_rf_model.feature_importances_

# scale byb max importance
feature_importance = feature_importance/feature_importance.max()
sorted_idx = np.argsort(feature_importance)
bar_pos = np.arange(feature_importance)
plt.barh(bar_pos, feature_importance[sorted_idx], align='center')
plt.yticks(bar_pos, wine_names[sorted_idx])
plt.xlabel('Variable Importance')
plt.show()
