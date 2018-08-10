import os
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl


def confusion_matrix(predicted, actual, threshold):
    if len(predicted) != len(actual):
        return -1
    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0

    for i in range(len(actual)):
        if actual[i] > 0.5:
            if predicted[i] > threshold:
                tp += 1.0
            else:
                fn += 1.0
        else:
            if predicted[i] < threshold:
                tn += 1.0
            else:
                fp += 1.0
    return [tp, fn, fp, tn]


# Read the dataset
dataset_file = os.path.join(os.getcwd(),
                            '../../datasets/sonar.all-data')
x_list = []
labels = []
with open(dataset_file, 'r') as f:
    for line in f.readlines():
        # split comma separated row
        row = line.strip().split(',')
        # Assign labels
        if row[-1] == 'M':
            labels.append(1.0)
        else:
            labels.append(0.0)
        # Remove label from row
        row.pop()
        # convert row to floats
        float_row = [float(num) for num in row]
        x_list.append(float_row)

# divide attribute matrix and label vector into training (2/3) and testing
# (1/3) data
indices = range(len(x_list))
xlist_test = [x_list[i] for i in indices if i % 3 == 0]
xlist_train = [x_list[i] for i in indices if i % 3 != 0]
labels_test = [labels[i] for i in indices if i % 3 == 0]
labels_train = [labels[i] for i in indices if i % 3 != 0]

# Make numpy array (using list of lists) to match input class
x_train = np.array(xlist_train)
y_train = np.array(labels_train)

x_test = np.array(xlist_test)
y_test = np.array(labels_test)

# Check the shape of training and testing data
print("Shape of xtrain (Training dataset): ", x_train.shape)
print("Shape of ytrain (Training labels): ", y_train.shape)

print("Shape of xtest (Testing dataset): ", x_test.shape)
print("Shape of ytest (Testing labels): ", y_test.shape)

# Train linear regression mode
rockvmine_model = linear_model.LinearRegression()
rockvmine_model.fit(x_train, y_train)

# generate predictions on in sample error
training_pred = rockvmine_model.predict(x_train)

print("Values predicted by model: ", training_pred[:5], training_pred[-6:-1])

# generate confusion matrix for predictions on training set (in-sample)
conf_mat_train = confusion_matrix(training_pred, y_train, 0.5)
# pick threshold value and generate confusion matrix entries
tp, fn, fp, tn = conf_mat_train
print('Confusion matrix for Training dataset\n'
      'tp={0},\nfn={1},\nfp={2},\ntn={3}'.format(tp, fn, fp, tn))

# Generate predictions on out of sample data
test_pred = rockvmine_model.predict(x_test)
conf_mat_test = confusion_matrix(test_pred, y_test, 0.5)
tp, fn, fp, tn = conf_mat_train
print('Confusion matrix for Testing dataset\n'
      'tp={0},\nfn={1},\nfp={2},\ntn={3}'.format(tp, fn, fp, tn))


# Generate ROC for in sample
fpr, tpr, thresholds = roc_curve(y_train, training_pred)
roc_auc = auc(fpr, tpr)
print('AUC for in-sample ROC curve: {0}'.format(roc_auc))

# Plot ROC
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area=%0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k-')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False positive rate')
pl.ylabel('True positive rate')
pl.title('In sample ROC rocks versus mine')
pl.legend(loc='lower right')
pl.show()

# Generate ROC for out sample
fpr, tpr, thresholds = roc_curve(y_test, test_pred)
roc_auc = auc(fpr, tpr)
print('AUC for out-sample ROC curve: {0}'.format(roc_auc))

# Plot ROC
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area=%0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k-')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False positive rate')
pl.ylabel('True positive rate')
pl.title('In sample ROC rocks versus mine')
pl.legend(loc='lower right')
pl.show()



