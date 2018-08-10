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
# Instantiate model
n_est = 2000
depth = 5
learn_rate = 0.005
max_features = 3
sub_samp = 0.5
abalone_gbm_model = ensemble.GradientBoostingRegressor(
    n_estimators=n_est, learning_rate=learn_rate, max_depth=depth,
    max_features=max_features, subsample=sub_samp, loss='ls')

# train
abalone_gbm_model.fit(x_train, y_train)
# compute mse on test set
ms_error = []
pred = abalone_gbm_model._staged_decision_function(x_test)
for p in pred:
    ms_error.append(mean_squared_error(y_test, p))

print("MSE: ", min(ms_error), ", ", ms_error.index(min(ms_error)))


# Plot training and test errors vs number of trees in ensemble
plt.figure()
plt.plot(range(1, n_est + 1), abalone_gbm_model.train_score_,
         label='Training Set MSE', linestyle=":")
plt.plot(range(1, n_est + 1), ms_error, label='Test Set MSE')
plt.legend(loc='upper right')
plt.xlabel('Number of Trees in Ensemble')
plt.ylabel('Mean Squared Error')
plt.show()

# Feature importance
feature_importance = abalone_gbm_model.feature_importances_
feature_importance = feature_importance/feature_importance.max()
sorted_idx = np.argsort(feature_importance)
bar_pos = np.arange(sorted_idx.shape[0])+ 0.5
plt.barh(bar_pos, feature_importance[sorted_idx], align='center')
plt.yticks(bar_pos, abalone_names[sorted_idx])
plt.xlabel('Variable importance')
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
plt.show()
