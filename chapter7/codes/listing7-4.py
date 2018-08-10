import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

dataset_file = os.path.join(os.getcwd(), '../../datasets/abalone.data')

x_list = []
labels = []

with open(dataset_file, 'r') as f:
    for line in f.readlines():
        row = line.strip().split(',')
        labels.append(float(row.pop()))
        x_list.append(row)


# Code three valued sex attribute as numeric
x_coded = []
for row in x_list:
    coded_sex = [0.0, 0.0]
    if row[0] == 'M':
        coded_sex[0] = 1.0
    elif row[0] == 'F':
        coded_sex[1] = 1.0

    num_row = [float(row[i]) for i in range(1, len(row))]
    row_coded = list(coded_sex) + num_row
    x_coded.append(row_coded)

# list of names
abalone_names = np.array(['Sex1', 'Sex2', 'Length', 'Diameter', 'Height',
                          'Whole weight', 'Shucked weight',
                          'Viscera weight', 'Shell weight', 'Rings'])
# number of rows an columns in x matrix
n_rows = len(x_coded)
n_cols = len(x_coded[0])

# code x and y in numpy arrays
x = np.array(x_coded)
y = np.array(labels)

# break into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,
                                                    random_state=531)

# train random forest at a range of ensemble sizes in order to see how to mse
#  changes
mse_oos = []
n_tree_list = range(50, 500, 10)
for i_trees in n_tree_list:
    depth = None
    max_feat = 4
    abalone_rf_model = ensemble.RandomForestRegressor(
        n_estimators=i_trees, max_depth=depth, max_features=max_feat,
        oob_score=False, random_state=531)

    abalone_rf_model.fit(x_train, y_train)
    # accumulate mse on test set
    pred = abalone_rf_model.predict(x_test)
    mse_oos.append(mean_squared_error(y_test, pred))

print("MSE: ", mse_oos[-1])

# Plot training and test errors vs number of trees in ensemble
plt.plot(n_tree_list, mse_oos)
plt.xlabel('Number of trees in ensemble')
plt.ylabel('Mean squared error')
plt.show()

# Feature importance
feature_importance = abalone_rf_model.feature_importances_
feature_importance = feature_importance/feature_importance.max()
sorted_idx = np.argsort(feature_importance)
bar_pos = np.arange(sorted_idx.shape[0])+ 0.5
plt.barh(bar_pos, feature_importance[sorted_idx], align='center')
plt.yticks(bar_pos, abalone_names[sorted_idx])
plt.xlabel('Variable importance')
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
plt.show()
