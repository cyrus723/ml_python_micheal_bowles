import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

dataset_file = os.path.join(os.getcwd(), '../../datasets/sonar.all-data')

xList = []

with open(dataset_file, 'r') as f:
    for line in f.readlines():
        row = line.strip().split(',')
        xList.append(row)

xNum = []
labels = []
for row in xList:
    lastCol = row.pop()
    if lastCol == "M":
        labels.append(1)
    else:
        labels.append(0)
    attrRow = [float(elt) for elt in row]
    xNum.append(attrRow)

# number of rows and columns in x matrix
nrows = len(xNum)
ncols = len(xNum[1])

# form x and y into numpy arrays and make up column names
X = np.array(xNum)
y = np.array(labels)

rocksVMinesNames = np.array(['V' + str(i) for i in range(ncols)])

# break into training and test sets.
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.30,
                                                random_state=531)
auc = []
nTreeList = range(50, 2000, 50)
for iTrees in nTreeList:
    depth = None
    maxFeat = 8 #try tweaking
    rocksVMinesRFModel = ensemble.RandomForestClassifier(
        n_estimators=iTrees, max_depth=depth, max_features=maxFeat,
        oob_score=False, random_state=531)
    rocksVMinesRFModel.fit(xTrain, yTrain)
    #Accumulate auc on test set
    prediction = rocksVMinesRFModel.predict_proba(xTest)
    aucCalc = roc_auc_score(yTest, prediction[:,1:2])
    auc.append(aucCalc)

print("AUC")
print(auc[-1])

# plot training and test errors vs number of trees in ensemble
plt.plot(nTreeList, auc)
plt.xlabel('Number of Trees in Ensemble')
plt.ylabel('Area Under ROC Curve - AUC')
# plt.ylim([0.0, 1.1*max(mseOob)])
plt.show()

# Plot feature importance
featureImportance = rocksVMinesRFModel.feature_importances_
# normalize by max importance
featureImportance = featureImportance / featureImportance.max()

# plot importance of top 30
idxSorted = np.argsort(featureImportance)[30:60]
idxTemp = np.argsort(featureImportance)[::-1]
print(idxTemp)
barPos = np.arange(idxSorted.shape[0]) + .5
plt.barh(barPos, featureImportance[idxSorted], align='center')
plt.yticks(barPos, rocksVMinesNames[idxSorted])
plt.xlabel('Variable Importance')
plt.show()
# plot best version of ROC curve
fpr, tpr, thresh = roc_curve(yTest, list(prediction[:, 1:2]))

ctClass = [i*0.01 for i in range(101)]
plt.plot(fpr, tpr, linewidth=2)
plt.plot(ctClass, ctClass, linestyle=':')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# pick some threshold values and calc confusion matrix for best predictions
# notice that GBM predictions don't fall in range of (0, 1) pick threshold
# values at 25th, 50th and 75th percentiles
idx25 = int(len(thresh) * 0.25)
idx50 = int(len(thresh) * 0.50)
idx75 = int(len(thresh) * 0.75)

# calculate total points, total positives and total negatives
totalPts = len(yTest)
P = sum(yTest)
N = totalPts - P
print('')
print('Confusion Matrices for Different Threshold Values')

# 25th
TP = tpr[idx25] * P; FN = P - TP; FP = fpr[idx25] * N; TN = N - FP
print('')
print('Threshold Value = ', thresh[idx25])
print('TP = ', TP/totalPts, 'FP = ', FP/totalPts)
print('FN = ', FN/totalPts, 'TN = ', TN/totalPts)

# 50th
TP = tpr[idx50] * P; FN = P - TP; FP = fpr[idx50] * N; TN = N - FP
print('')
print('Threshold Value = ', thresh[idx50])
print('TP = ', TP/totalPts, 'FP = ', FP/totalPts)
print('FN = ', FN/totalPts, 'TN = ', TN/totalPts)

# 75th
TP = tpr[idx75] * P; FN = P - TP; FP = fpr[idx75] * N; TN = N - FP
print('')
print('Threshold Value = ', thresh[idx75])
print('TP = ', TP/totalPts, 'FP = ', FP/totalPts)
print('FN = ', FN/totalPts, 'TN = ', TN/totalPts)