import os
import numpy as np
from math import sqrt
from sklearn.linear_model import enet_path
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def getNtileBoundaries(data, ntiles):
    percentBdry = []
    for i in range(ntiles+1):
        percentBdry.append(np.percentile(data, i*(100)/ntiles))
    return percentBdry


x_list = []
labels = []
# Open the data set
datafile = os.path.join(os.getcwd(),
                        '../../datasets/sonar.all-data')
f = open(datafile, 'r')

for line in f.readlines():
    row = line.strip().split(',')
    x_list.append(row)

x_num = []
labels = []

for row in x_list:
    last_col = row.pop()
    if last_col == 'M':
        labels.append(1.0)
    else:
        labels.append(0.0)
    attr_row = [float(elt) for elt in row]
    x_num.append(attr_row)

# Number of rows and columns
n_row = len(x_num)
n_col = len(x_num[1])

alpha = 1.0

# calculate means and variances
x_means = []
x_sd = []
for i in range(n_col):
    col = [x_num[j][i] for j in range(n_row)]
    mean = sum(col)/n_row
    x_means.append(mean)
    col_diff = [(x_num[j][i] - mean) for j in range(n_row)]
    sum_sq = sum([col_diff[i] ** 2 for i in range(n_row)])
    stdev = sqrt(sum_sq/n_row)
    x_sd.append(stdev)

# Use calculate mean and standard deviation to normalize x_num
x_norm = []
for i in range(n_row):
    row_norm = [(x_num[i][j] - x_means[j])/x_sd[j] for j in range(n_col)]
    x_norm.append(row_norm)

# Normalize labels to center
mean_label = sum(labels)/n_row
sd_label = sqrt(sum([(labels[i] - mean_label) * (labels[i] - mean_label) \
                     for i in range(n_row)])/n_row)
labels_norm = [(labels[i] -mean_label)/sd_label]

# number of cross-validation folds
nx_val = 10

for ix_val in range(nx_val):
    # define test and training data
    idx_test = [a for a in range(n_row) if a % nx_val == ix_val % nx_val]
    idx_train = [a for a in range(n_row) if a % nx_val != ix_val % nx_val]

    # Define test and training attributes and label sets
    x_train = np.array([x_norm[r] for r in idx_train])
    x_test = np.array([x_norm[r] for r in idx_test])
    label_train = np.array([labels_norm[r] for r in idx_train])
    label_test = np.array([labels_norm[r] for r in idx_test])

    alphas, coefs, _ = enet_path(x_train, label_train, l1_ratio=0.8,
                                 fit_intercept=False, return_models=False)
    # apply coefs to test data to produce predictions and accumulate
    if ix_val == 0:
        pred = np.dot(x_test, coefs)
        y_out = label_test
    else:
        # accumulate predictions
        y_temp = np.array(y_out)
        y_out = np.concatenate((y_temp, label_test), axis=0)

        # accumulate predictions
        pred_temp = np.array(pred)
        pred = np.concatenate((pred_temp, np.dot(x_test, coefs)), axis=0)

# calculate misclassification error
miss_class_rate = []
_, n_pred = pred.shape
for i_pred in range(1, n_pred):
    pred_list = list(pred[:, i_pred])
    err_cnt = 0.0
    for i_row in range(n_row):
        if pred_list[i_row] < 0.0 and y_out[i_row] >= 0.0:
            err_cnt += 1.0
        elif pred_list[i_row] >= 0.0 and y_out[i_row] < 0.0:
            err_cnt += 1.0
    miss_class_rate.append(err_cnt/n_row)

# Find minimum point for plot and for point
min_err = min(miss_class_rate)
idx_min = miss_class_rate.index(min_err)
plot_alphas = list(alphas[1:len(alphas)])

plt.figure()
plt.plot(plot_alphas, miss_class_rate,
         label='Misclassification Error Across Folds', linestyle='--')
plt.axvline(plot_alphas[idx_min], linestyle='--',
            label='CV estimate of best alpha')
plt.legend()
plt.semilogx()
ax = plt.gca()
ax.invert_xaxis()
plt.xlabel('alpha')
plt.ylabel('Misclassification error')
plt.axis('tight')
plt.show()

# Calculate AUC
idx_pos = [i for i in range(n_row) if y_out[i] > 0.0]
y_out_bin = [0] * n_row
for i in idx_pos:
    y_out_bin[i] = 1

auc = []
for i_pred in range(1, n_pred):
    pred_list = list(pred[:, i_pred])
    auc_calc = roc_auc_score(y_out_bin, pred_list)
    auc.append(auc_calc)

max_auc = max(auc)
idx_max = auc.index(max_auc)

plt.figure()
plt.plot(plot_alphas, auc, label='AUC across folds', linewidth=2)
plt.axvline(plot_alphas[idx_max], linestyle='--',
            label='CV Estimate of best alpha')
plt.legend()
plt.semilogx()
ax = plt.gca()
ax.invert_xaxis()
plt.xlabel('alpha')
plt.ylabel('Area under the ROC curve')
plt.axis('tight')
plt.show()

# Plot best version of ROC curve
fpr, tpr, thresh = roc_curve(y_out_bin, list(pred[:, idx_max]))
ct_class = [i * 0.01 for i in range(101)]

plt.plot(fpr, tpr, linewidth=2)
plt.plot(ct_class, ct_class, linestyle=':')
plt.xlabel('False positive  rate')
plt.ylabel('True positive rate')
plt.show()

print('Best value of misclassification error = ', miss_class_rate[idx_min])
print('Best alpha of misclassification error = ', plot_alphas[idx_min])
print('')
print('Best value for AUC = ', auc[idx_max])
print('Best alpha for AUC = ', plot_alphas[idx_max])
print('')
print('Confusion matrices for different threshold values')

# pick some points along the curve to print. There are 208 points.
# The extremes aren't useful
# Sample at 52, 104 and 156. Use the calculated values of tpr and fpr
# along with definitions and threshold values.
# Some nomenclature (e.g. see wikipedia "receiver operating curve")
# P = Positive cases
P = len(idx_pos)
# N = Negative cases
N = n_row - P
# TP = True positives = tpr * P
TP = tpr[52] * P
# FN = False negatives = P - TP
FN = P - TP
# FP = False positives = fpr * N
FP = fpr[52] * N
# TN = True negatives = N - FP
TN = N - FP
print('Threshold Value = ', thresh[52])
print('TP = ', TP, 'FP = ', FP)
print('FN = ', FN, 'TN = ', TN)
TP = tpr[104] * P
FN = P - TP
FP = fpr[104] * N
TN = N - FP
print('Threshold Value = ', thresh[104])
print('TP = ', TP, 'FP = ', FP)
print('FN = ', FN, 'TN = ', TN)
TP = tpr[156] * P
FN = P - TP
FP = fpr[156] * N
TN = N - FP
print('Threshold Value = ', thresh[156])
print('TP = ', TP, 'FP = ', FP)
print('FN = ', FN, 'TN = ', TN)