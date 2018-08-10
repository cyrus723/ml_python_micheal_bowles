import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


dataset_file = os.path.join(os.getcwd(), '../../datasets/glass.data')

xList = []

with open(dataset_file, 'r') as f:
    for line in f.readlines():
        # split on comma
        row = line.strip().split(",")
        xList.append(row)

glassNames = np.array(['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe',
                       'Type'])

# Separate attributes and labels
xNum = []
labels = []
for row in xList:
    labels.append(row.pop())
    l = len(row)
    # eliminate ID
    attrRow = [float(row[i]) for i in range(1, l)]
    xNum.append(attrRow)
# number of rows and columns in x matrix
nrows = len(xNum)
ncols = len(xNum[1])

# Labels are integers from 1 to 7 with no examples of 4. gb requires
# consecutive integers starting at 0
newLabels = []
labelSet = set(labels)
labelList = list(labelSet)
labelList.sort()
nlabels = len(labelList)
for l in labels:
    index = labelList.index(l)
newLabels.append(index)

# stratified sampling by labels.
xTemp = [xNum[i] for i in range(nrows) if newLabels[i] == 0]
yTemp = [newLabels[i] for i in range(nrows) if newLabels[i] == 0]
xTrain, xTest, yTrain, yTest = train_test_split(xTemp, yTemp, test_size=0.30,
                                                random_state=531)
for iLabel in range(1, len(labelList)):
    # segregate x and y according to labels
    xTemp = [xNum[i] for i in range(nrows) if newLabels[i] == iLabel]
    yTemp = [newLabels[i] for i in range(nrows)
             if newLabels[i] == iLabel]
    # form train and test sets on segregated subset of examples
    xTrainTemp, xTestTemp, yTrainTemp, yTestTemp = train_test_split(
        xTemp, yTemp, test_size=0.30, random_state=531)
    # accumulate
    xTrain = np.append(xTrain, xTrainTemp, axis=0)
    xTest = np.append(xTest, xTestTemp, axis=0)
    yTrain = np.append(yTrain, yTrainTemp, axis=0)
    yTest = np.append(yTest, yTestTemp, axis=0)

missCLassError = []
nTreeList = range(50, 2000, 50)
for iTrees in nTreeList:
    depth = None
    maxFeat = 4  # try tweaking
    glassRFModel = ensemble.RandomForestClassifier(n_estimators=iTrees,
                                                   max_depth=depth,
                                                   max_features=maxFeat,
                                                   oob_score=False,
                                                   random_state=531)
    glassRFModel.fit(xTrain, yTrain)
    # Accumulate auc on test set
    prediction = glassRFModel.predict(xTest)
    correct = accuracy_score(yTest, prediction)
    missCLassError.append(1.0 - correct)

print("Missclassification Error")
print(missCLassError[-1])

# generate confusion matrix
pList = prediction.tolist()
confusionMat = confusion_matrix(yTest, pList)
print('')
print("Confusion Matrix")
print(confusionMat)

# plot training and test errors vs number of trees in ensemble
plt.plot(nTreeList, missCLassError)
plt.xlabel('Number of Trees in Ensemble')
plt.ylabel('Missclassification Error Rate')
# plt.ylim([0.0, 1.1*max(mseOob)])
plt.show()

# Plot feature importance
featureImportance = glassRFModel.feature_importances_

# normalize by max importance
featureImportance = featureImportance / featureImportance.max()

# plot variable importance
idxSorted = np.argsort(featureImportance)
barPos = np.arange(idxSorted.shape[0]) + .5
plt.barh(barPos, featureImportance[idxSorted], align='center')
plt.yticks(barPos, glassNames[idxSorted])
plt.xlabel('Variable Importance')
plt.show()
