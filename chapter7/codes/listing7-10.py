import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix


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

# instantiate model
nEst = 500
depth = 3
learnRate = 0.003
maxFeatures = 3
subSamp = 0.5
glassGBMModel = ensemble.GradientBoostingClassifier(
    n_estimators=nEst, max_depth=depth, learning_rate=learnRate,
    max_features=maxFeatures, subsample=subSamp)

# train
glassGBMModel.fit(xTrain, yTrain)
# compute auc on test set as function of ensemble size
missClassError = []
missClassBest = 1.0
predictions = glassGBMModel.staged_decision_function(xTest)
for p in predictions:
    missClass = 0
    for i in range(len(p)):
        listP = p[i].tolist()
    if listP.index(max(listP)) != yTest[i]:
        missClass += 1
    missClass = float(missClass) / len(p)
    missClassError.append(missClass)
    # capture best predictions
    if missClass < missClassBest:
        missClassBest = missClass
        pBest = p

idxBest = missClassError.index(min(missClassError))

# print best values
print("Best Missclassification Error" )
print(missClassBest)
print("Number of Trees for Best Missclassification Error")
print(idxBest)

# plot training deviance and test auc's vs number of trees in ensemble
missClassError = [100*mce for mce in missClassError]
plt.figure()
plt.plot(range(1, nEst + 1), glassGBMModel.train_score_,
          label='Training Set Deviance', linestyle=":")
plt.plot(range(1, nEst + 1), missClassError, label='Test Set Error')
plt.legend(loc='upper right')
plt.xlabel('Number of Trees in Ensemble')
plt.ylabel('Deviance / Classification Error')
plt.show()

# Plot feature importance
featureImportance = glassGBMModel.feature_importances_
# normalize by max importance
featureImportance = featureImportance / featureImportance.max()

# plot variable importance
idxSorted = np.argsort(featureImportance)
barPos = np.arange(idxSorted.shape[0]) + .5
plt.barh(barPos, featureImportance[idxSorted], align='center')
plt.yticks(barPos, glassNames[idxSorted])
plt.xlabel('Variable Importance')
plt.show()

# generate confusion matrix for best prediction.
pBestList = pBest.tolist()
bestPrediction = [r.index(max(r)) for r in pBestList]
confusionMat = confusion_matrix(yTest, bestPrediction)
print('')
print("Confusion Matrix")
print(confusionMat)
