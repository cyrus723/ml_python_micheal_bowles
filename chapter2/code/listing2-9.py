from math import sqrt
import matplotlib.pyplot as plt
import os
import pandas as pd
from random import uniform

datafile = os.path.join(os.getcwd(),
                        '../../datasets/sonar.all-data')

rocks_vs_mines = pd.read_csv(datafile, header=None, prefix='V')

data_row2 = rocks_vs_mines.iloc[1, 0:60]
data_row3 = rocks_vs_mines.iloc[2, 0:60]
data_row21 = rocks_vs_mines.iloc[21, 0:60]

mean2 = 0.0
mean3 = 0.0
mean21 = 0.0

numElt = len(data_row2)
for i in range(numElt):
    mean2 += data_row2[i]/numElt
    mean3 += data_row3[i]/numElt
    mean21 += data_row21[i]/numElt

var2 = 0.0
var3 = 0.0
var21 = 0.0
for i in range(numElt):
    var2 += (data_row2[i] - mean2) * (data_row2[i] - mean2) / numElt
    var3 += (data_row3[i] - mean3) * (data_row3[i] - mean3) / numElt
    var21 += (data_row21[i] - mean21) * (data_row21[i] - mean21) / numElt

corr23 = 0.0; corr221 = 0.0
for i in range(numElt):
    corr23 += (data_row2[i] - mean2) * (data_row3[i] - mean3) / (sqrt(var2*var3) * numElt)
    corr221 += (data_row2[i] - mean2) * (data_row21[i] - mean21) / (sqrt(
        var2*var21) * numElt)

print("Correlation between attr2 and attr3={}".format(corr23))
print("Correlation between attr2 and attr21={}".format(corr221))
