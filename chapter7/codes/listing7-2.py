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

# train gradient boosting model to miniize mean square error
n_est = 2000
depth = 7
learn_rate = 0.01
sub_samp = 0.5

wine_gbm_model = ensemble.GradientBoostingRegressor(
    n_estimators=n_est, max_depth=depth, learning_rate=learn_rate,
    subsample=sub_samp, loss='ls')

wine_gbm_model.fit(x_train, y_train)

# compute mse on test set
ms_error = []
pred = wine_gbm_model.staged_predict(x_test)
for p in pred:
    ms_error.append(mean_squared_error(y_test, p))

print('MSE: ', min(ms_error), ms_error.index(ms_error))

# plot training and test errors vs number of trees in ensemble
plt.figure()
plt.plot(range(1, n_est+1), wine_gbm_model.train_score_,
         label='Training set MSE')
plt.plot(range(1, n_est+1), ms_error, label='Test set MSE')
plt.legend(loc='upper right')
plt.xlabel('Number of trees in Ensemble')
plt.ylabel('Mean squared error')
plt.show()

# Plot feature importance
feature_importance = wine_gbm_model.feature_importances_

# normalize by max importance
feature_importance = feature_importance/feature_importance.max()
idx_sorted = np.argsort(feature_importance)
bar_pos = np.arange(idx_sorted.shape[0]) + 0.5
plt.barh(bar_pos, feature_importance[idx_sorted], align='center')
plt.yticks(bar_pos, wine_names[idx_sorted])
plt.xlabel('Variable importance')
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
plt.show()
