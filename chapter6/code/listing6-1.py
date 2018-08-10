import os
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals.six import StringIO
from math import sqrt
import matplotlib.pyplot as plt

# read the data

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

wine_tree = DecisionTreeRegressor(max_depth=3)
wine_tree.fit(x_list, labels)

with open('wine_tree.dot', 'w') as f:
    f = tree.export_graphviz(wine_tree, out_file=f)
# Note: The code above exports the trained tree info to a Graphviz "dot" file.
# Drawing the graph requires installing GraphViz and the running the
# following on the command line
# dot -Tpng wineTree.dot -o wineTree.png
# In Windows, you can also open the .dot file in the GraphViz gui (GVedit.exe)]
